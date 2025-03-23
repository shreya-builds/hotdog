import streamlit as st
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
from typing import List, Dict

# Initialize models with minimal set
@st.cache_resource
def load_models():
    models = {}
    try:
        models['text_gen'] = pipeline("text2text-generation", model="facebook/bart-large-cnn")
        models['embedding'] = SentenceTransformer('all-MiniLM-L6-v2')
        models['plot'] = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        raise
    return models

models = load_models()

# TMDB API configuration
TMDB_API_KEY = "2d4a65d3f3fca1929295365d4fbe9745"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

@st.cache_data
def load_movie_data():
    """Load and process movie data from TMDB dataset"""
    try:
        return fetch_tmdb_data()
    except Exception as e:
        st.warning(f"Using backup dataset due to API error: {str(e)}")
        return get_sample_dataset()

def get_sample_dataset():
    """Fallback sample dataset"""
    return pd.DataFrame({
        'title': [
            'The Shawshank Redemption', 'The Godfather', 'Pulp Fiction', 'The Dark Knight'
        ],
        'plot': [
            'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
            'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
            'The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.',
            'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.'
        ],
        'genres': [
            'Drama', 'Crime,Drama', 'Crime,Drama', 'Action,Crime,Drama'
        ],
        'rating': [9.3, 9.2, 8.9, 9.0],
        'year': ['1994', '1972', '1994', '2008']
    })

def fetch_tmdb_data():
    """Fetch movie data from TMDB API"""
    base_url = "https://api.themoviedb.org/3"
    movies = []
    
    try:
        response = requests.get(
            f"{base_url}/movie/popular",
            params={
                "api_key": TMDB_API_KEY,
                "page": 1,
                "language": "en-US"
            }
        )
        response.raise_for_status()
        data = response.json()
        
        for movie in data["results"][:20]:  # Limit to 20 movies
            movies.append({
                "title": movie["title"],
                "plot": movie["overview"],
                "genres": ",".join([str(g) for g in movie["genre_ids"]]),
                "rating": movie["vote_average"],
                "year": movie["release_date"][:4] if movie["release_date"] else None
            })
        
        return pd.DataFrame(movies)
    
    except Exception as e:
        st.error(f"Error fetching data from TMDB: {str(e)}")
        raise

# Streamlit Interface
st.title("🎬 Movie Recommendation System")

# Main recommendation interface
movie_input = st.text_input("Enter a movie you like:")

if movie_input:
    movies_df = load_movie_data()
    
    # Find the movie in our database
    movie_match = movies_df[movies_df['title'].str.lower() == movie_input.lower()]
    
    if not movie_match.empty:
        movie = movie_match.iloc[0]
        
        # Get embeddings for all movies
        all_embeddings = models['embedding'].encode(movies_df['plot'].tolist())
        input_embedding = models['embedding'].encode(movie['plot'])
        
        # Calculate similarities
        similarities = np.dot(all_embeddings, input_embedding)
        
        # Get top 3 similar movies
        similar_indices = np.argsort(similarities)[-4:-1][::-1]
        
        st.subheader("Similar Movies")
        for idx in similar_indices:
            st.write(f"**{movies_df.iloc[idx]['title']} ({movies_df.iloc[idx]['year']})**")
            st.write(f"Rating: {movies_df.iloc[idx]['rating']}/10")
            st.write(f"Genres: {movies_df.iloc[idx]['genres']}")
            st.write(movies_df.iloc[idx]['plot'])
            st.write("---")
    else:
        st.error("Movie not found in database. Please try another movie.") 