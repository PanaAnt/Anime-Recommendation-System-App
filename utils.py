import pandas as pd
import os
from dotenv import load_dotenv
import requests
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Database connection details from environment variables
DATABASE_TYPE = 'postgresql'
DBAPI = 'psycopg2'
ENDPOINT = os.getenv('DB_HOST')
USER = os.getenv('DB_USER')
PASSWORD = os.getenv('DB_PASSWORD')
PORT = os.getenv('DB_PORT')
DATABASE = os.getenv('DB_NAME')

engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{ENDPOINT}:{PORT}/{DATABASE}")

# Function to fetch data from PostgreSQL and preprocess it
def fetch_and_preprocess_data():
    # Step 1: Fetch data from PostgreSQL
    query = "SELECT * FROM anime_info;"
    anime_df = pd.read_sql(query, engine)
    
    # Step 2: Preprocessing
    anime_df['type'] = anime_df['type'].fillna('Unknown')
    anime_df['status'] = anime_df['status'].fillna('Unknown')
    anime_df['scored_by'] = anime_df['scored_by'].fillna(0)
    anime_df['favorites'] = anime_df['favorites'].fillna(0)
    anime_df['episodes'] = anime_df['episodes'].fillna(0)
    anime_df['start_date'] = anime_df['start_date'].fillna(pd.Timestamp('0001-01-01'))
    anime_df['end_date'] = anime_df['end_date'].fillna(pd.Timestamp('0001-01-01'))
    anime_df = anime_df.dropna(subset=['anime_id', 'title'])
    anime_df['score'] = anime_df['score'].fillna(0)
    anime_df['source'] = anime_df['source'].fillna('Unknown')
    anime_df['rating'] = anime_df['rating'].fillna('N/A')

    # Optionally drop columns that are not needed
    anime_df_cleaned = anime_df.drop(columns=['url', 'title_english', 'licensors', 'studios', 'broadcast_day', 'total_duration', 'producers', 'trailer_url', 'background'])
    
    anime_df_cleaned = pd.get_dummies(anime_df_cleaned, columns=['type', 'status', 'rating'], drop_first=True)

    # Step 3: Check if 'genres', 'themes', and 'demographics' exist in the DataFrame
    for column in ['genres', 'themes', 'demographics']:
        if column not in anime_df_cleaned.columns:
            anime_df_cleaned[column] = ''  # Create an empty column if missing

    # Function to clean the genres, themes, and demographics strings
    def clean_labels(column):
        column = column.str.replace(r'\[|\]', '', regex=True)
        column = column.str.strip().str.lower().str.replace("  ", " ") 
        return column

    # Apply cleaning function if the columns exist
    anime_df_cleaned['genres'] = clean_labels(anime_df_cleaned['genres'])
    anime_df_cleaned['themes'] = clean_labels(anime_df_cleaned['themes'])
    anime_df_cleaned['demographics'] = clean_labels(anime_df_cleaned['demographics'])

    # One-hot encode the genre, theme, and demographic data
    genres_split = anime_df_cleaned['genres'].str.get_dummies(sep=', ').add_prefix('genres_')
    themes_split = anime_df_cleaned['themes'].str.get_dummies(sep=', ').add_prefix('themes_')
    demographics_split = anime_df_cleaned['demographics'].str.get_dummies(sep=', ').add_prefix('demographics_')

    anime_df_cleaned = pd.concat([anime_df_cleaned, genres_split, themes_split, demographics_split], axis=1)

    # Drop the original columns after one-hot encoding
    anime_df_cleaned = anime_df_cleaned.drop(columns=['genres', 'themes', 'demographics'])

    # Scaling numerical columns
    numerical_cols = ['score', 'scored_by', 'favorites', 'episodes']
    anime_df_cleaned[numerical_cols] = anime_df_cleaned[numerical_cols].replace('N/A', pd.NA)
    anime_df_cleaned[numerical_cols] = anime_df_cleaned[numerical_cols].apply(pd.to_numeric, errors='coerce')
    anime_df_cleaned = anime_df_cleaned.fillna(0)
    scaler = MinMaxScaler()
    anime_df_cleaned[numerical_cols] = scaler.fit_transform(anime_df_cleaned[numerical_cols])

    return anime_df_cleaned

# Function to compute cosine similarity
def compute_similarity_matrix(anime_df_cleaned):
    genre_theme_columns = [col for col in anime_df_cleaned.columns if 'themes_' in col or 'demographics_' in col or 'genres_' in col or 'type_' in col or 'rating_' in col]
    anime_one_hot = anime_df_cleaned[genre_theme_columns]
    return cosine_similarity(anime_one_hot)

# Recommendation function based on anime title
def recommend_anime_by_id(anime_id, anime_df, similarity_matrix):
    # Find the index of the anime by its ID
    anime_index = anime_df[anime_df['anime_id'] == anime_id].index[0]

    # Get similarity scores for this anime
    similarity_scores = list(enumerate(similarity_matrix[anime_index]))

    # Sort the anime based on similarity scores in descending order
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get the IDs of the most similar anime (excluding the selected anime itself)
    similar_anime_indices = [i[0] for i in similarity_scores[1:21]]

    # Return the anime IDs of the most similar anime
    similar_anime_ids = anime_df.iloc[similar_anime_indices]['anime_id'].values
    return similar_anime_ids

def get_top_anime():
    url = "https://api.jikan.moe/v4/top/anime"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        top_anime_list = data['data'][:10]  # Get only the top 10 anime
        return top_anime_list
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from JikanAPI: {e}")
        return []

