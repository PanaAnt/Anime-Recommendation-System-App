from flask import Flask, render_template, request, jsonify
import requests
from flask_sqlalchemy import SQLAlchemy
from config import Config
from utils import fetch_and_preprocess_data, compute_similarity_matrix, recommend_anime_by_id, get_top_anime

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)

# Database Columns in PostgreSQL
class Anime(db.Model):
    __tablename__ = 'anime_info'
    anime_id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String, nullable=False)
    type = db.Column(db.String)
    score = db.Column(db.Float)
    scored_by = db.Column(db.Integer)
    status = db.Column(db.String)
    episodes = db.Column(db.Integer)
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)
    source = db.Column(db.String)
    favorites = db.Column(db.Integer)
    total_duration = db.Column(db.String)
    rating = db.Column(db.String)
    start_year = db.Column(db.Integer)
    start_season = db.Column(db.String)
    broadcast_day = db.Column(db.String)
    genres = db.Column(db.String)
    themes = db.Column(db.String)
    demographics = db.Column(db.String)
    studios = db.Column(db.String)
    producers = db.Column(db.String)
    licensors = db.Column(db.String)
    synopsis = db.Column(db.Text)
    background = db.Column(db.Text)
    main_picture = db.Column(db.String)
    url = db.Column(db.String)
    trailer_url = db.Column(db.String)
    title_english = db.Column(db.String)

# Load the anime data and compute the similarity matrix
anime_df_cleaned = fetch_and_preprocess_data()
similarity_matrix = compute_similarity_matrix(anime_df_cleaned)

@app.route('/')
def index():
    # Get top 10 anime from Jikan API
    response = requests.get('https://api.jikan.moe/v4/top/anime')
    top_anime_data = response.json()

    # Extracting relevant details for the top 10 anime
    top_anime = [
        {
            'title': anime['title'],
            'image_url': anime['images']['jpg']['image_url'],  # This should be the correct image URL
            'score': anime.get('score', 'N/A'),  # Fallback to 'N/A' if no score
            'url': anime['url']
        }
        for anime in top_anime_data['data'][:10]
    ]

    # Checking if fetched correctly
    print(top_anime)

    return render_template('index.html', top_anime=top_anime)

# Route for fetching title suggestions (for autocomplete)
@app.route('/title_suggestions')
def title_suggestions():
    query = request.args.get('query', '')
    if query:
        suggestions = Anime.query.filter(Anime.title.ilike(f"%{query}%")).limit(5).all()
        return jsonify([{"anime_id": anime.anime_id, "title": anime.title} for anime in suggestions])
    return jsonify([])

# Route to handle anime search and recommendations
@app.route('/recommendations', methods=['POST'])
def recommendations():
    # Converting anime_id to a Python int
    anime_id = int(request.form.get('anime_id'))  

    # Finding the anime by its ID
    anime = Anime.query.filter(Anime.anime_id == anime_id).first()

    if anime:
        # Getting recommendations using the anime_id for content-based filtering technique I went with
        recommended_anime = recommend_anime_by_id(anime.anime_id, anime_df_cleaned, similarity_matrix)

        if recommended_anime is not None and len(recommended_anime) > 0:
            # Preparing recommendations in a list of dictionaries format
            recommendations_data = []
            for rec_id in recommended_anime:
                # Ensuring that rec_id is also a Python int
                rec_anime = Anime.query.filter(Anime.anime_id == int(rec_id)).first()
                if rec_anime:
                    recommendations_data.append({
                        "title": rec_anime.title,
                        "main_picture": rec_anime.main_picture,
                        "score": rec_anime.score,
                        "url": rec_anime.url
                    })
            return render_template('recommendations.html', anime=anime, recommendations=recommendations_data)
        else:
            return render_template('index.html', error=f"No recommendations found for '{anime.title}'")
    else:
        return render_template('index.html', error="Anime not found.")

if __name__ == '__main__':
    app.run(debug=True)