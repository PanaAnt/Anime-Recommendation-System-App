import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine

# Define connection to your PostgreSQL database
DATABASE_TYPE = 'postgresql'
DBAPI = 'psycopg2'
ENDPOINT = 'localhost'  # Change if using a remote server
USER = 'postgres'
PASSWORD = 'Alympana11'
PORT = 5432
DATABASE = 'anime_manga_recommendations'

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
    anime_df_cleaned = anime_df.drop(columns=['url', 'title_english', 'licensors', 'studios', 'broadcast_day','total_duration', 'producers', 'trailer_url', 'background'])
    
    anime_df_cleaned = pd.get_dummies(anime_df_cleaned, columns=['type', 'status', 'rating'], drop_first=True) 

    # Function to clean the genres, themes, and demographics strings
    def clean_labels(column):
        column = column.str.replace(r'\[|\]', '', regex=True)  
        column = column.str.strip().str.lower().str.replace("  ", " ") 
        return column

    # Apply cleaning function
    anime_df_cleaned['genres'] = clean_labels(anime_df_cleaned['genres'])
    anime_df_cleaned['themes'] = clean_labels(anime_df_cleaned['themes'])
    anime_df_cleaned['demographics'] = clean_labels(anime_df_cleaned['demographics'])

    genres_split = anime_df_cleaned['genres'].str.get_dummies(sep=', ').add_prefix('genres_')
    themes_split = anime_df_cleaned['themes'].str.get_dummies(sep=', ').add_prefix('themes_')
    demographics_split = anime_df_cleaned['demographics'].str.get_dummies(sep=', ').add_prefix('demographics_')

    anime_df_cleaned = pd.concat([anime_df_cleaned, genres_split, themes_split, demographics_split], axis=1)
    anime_df_cleaned = anime_df_cleaned.drop(columns=['genres', 'themes', 'demographics'])

    # Scaling numerical columns
    numerical_cols = ['score', 'scored_by', 'favorites', 'episodes']
    anime_df_cleaned[numerical_cols] = anime_df_cleaned[numerical_cols].replace('N/A', pd.NA)
    anime_df_cleaned[numerical_cols] = anime_df_cleaned[numerical_cols].apply(pd.to_numeric, errors='coerce')
    anime_df_cleaned = anime_df_cleaned.fillna(0)
    scaler = MinMaxScaler()
    anime_df_cleaned[numerical_cols] = scaler.fit_transform(anime_df_cleaned[numerical_cols])

    return anime_df_cleaned

anime_df_cleaned = fetch_and_preprocess_data()

# Extract relevant columns and compute cosine similarity
genre_theme_columns = [col for col in anime_df_cleaned.columns if 'themes_' in col or 'demographics_' in col or 'genres_' in col or 'type_' in col or 'rating_' in col]
anime_one_hot = anime_df_cleaned[genre_theme_columns]
similarity_matrix_sample = cosine_similarity(anime_one_hot)

# Recommendation function
def recommend_anime_by_title(anime_title, anime_df, similarity_matrix):
    if anime_title not in anime_df['title'].values:
        return f"Anime titled '{anime_title}' not found in the dataset."
    anime_index = anime_df[anime_df['title'] == anime_title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[anime_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_anime_indices = [i[0] for i in similarity_scores[1:21]]
    similar_anime_titles = anime_df.iloc[similar_anime_indices]['title'].values
    return similar_anime_titles

# Example usage
title = anime_df_cleaned['title'].values[11]
recommended_anime = recommend_anime_by_title(title, anime_df_cleaned, similarity_matrix_sample)
print(f"If you like '{title}' then you will like:\n", recommended_anime)