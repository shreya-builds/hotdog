import streamlit as st
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModel, VisionTextDualEncoderModel, AutoProcessor
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from typing import List, Dict

# Initialize models
@st.cache_resource
def load_models():
    return {
        'sentiment': pipeline("sentiment-analysis"),
        'text_gen': pipeline("text2text-generation", model="facebook/bart-large-cnn"),
        'embedding': SentenceTransformer('all-MiniLM-L6-v2'),
        'clip': VisionTextDualEncoderModel.from_pretrained("openai/clip-vit-base-patch32"),
        'processor': AutoProcessor.from_pretrained("openai/clip-vit-base-patch32"),
        'genre': pipeline("text-classification", model="facebook/bart-large-mnli"),
        'plot': pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    }

models = load_models()

# Multilingual support
multilingual_model = pipeline("translation", model="facebook/mbart-large-50-many-to-many-mmt")

# TMDB API configuration
TMDB_API_KEY = "2d4a65d3f3fca1929295365d4fbe9745"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

@st.cache_data
def load_movie_data():
    """Load and process movie data from TMDB dataset"""
    try:
        # Try to fetch data from TMDB API
        return fetch_tmdb_data()
    except Exception as e:
        st.warning(f"Using backup dataset due to API error: {str(e)}")
        # Fallback to sample dataset if API fails
        return get_sample_dataset()

def get_sample_dataset():
    """Fallback sample dataset"""
    return pd.DataFrame({
        'title': [
            'The Shawshank Redemption', 'The Godfather', 'Pulp Fiction', 'The Dark Knight', 
            'Inception', 'The Matrix', 'Interstellar', 'The Silence of the Lambs'
        ],
        'plot': [
            'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
            'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
            'The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.',
            'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.',
            'A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.',
            'A computer programmer discovers that reality as he knows it is a simulation created by machines, and joins a rebellion to break free.',
            'A team of explorers travel through a wormhole in space in an attempt to ensure humanity\'s survival.',
            'A young F.B.I. cadet must receive the help of an incarcerated and manipulative cannibal killer to help catch another serial killer.'
        ],
        'genres': [
            'Drama', 'Crime,Drama', 'Crime,Drama', 'Action,Crime,Drama',
            'Action,Adventure,Sci-Fi', 'Action,Sci-Fi', 'Adventure,Drama,Sci-Fi', 'Crime,Drama,Thriller'
        ],
        'poster_url': [
            'https://image.tmdb.org/t/p/w500/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg',
            'https://image.tmdb.org/t/p/w500/3bhkrj58Vtu7enYsRolD1fZdja1.jpg',
            'https://image.tmdb.org/t/p/w500/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg',
            'https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg',
            'https://image.tmdb.org/t/p/w500/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg',
            'https://image.tmdb.org/t/p/w500/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg',
            'https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg',
            'https://image.tmdb.org/t/p/w500/rplLJ2hPcOQmkFhTqUte0MkEaO2.jpg'
        ],
        'rating': [9.3, 9.2, 8.9, 9.0, 8.8, 8.7, 8.6, 8.6],
        'year': ['1994', '1972', '1994', '2008', '2010', '1999', '2014', '1991'],
        'runtime': [142, 175, 154, 152, 148, 136, 169, 118],
        'popularity': [83.2, 70.1, 68.3, 72.5, 74.8, 69.2, 71.4, 65.9]
    })

def fetch_tmdb_data():
    """Fetch movie data from TMDB API"""
    base_url = "https://api.themoviedb.org/3"
    movies = []
    
    try:
        # Fetch popular movies
        for page in range(1, 6):  # Get 100 movies (20 per page)
            response = requests.get(
                f"{base_url}/movie/popular",
                params={
                    "api_key": TMDB_API_KEY,
                    "page": page,
                    "language": "en-US"
                }
            )
            response.raise_for_status()  # Raise exception for bad status codes
            data = response.json()
            
            # Fetch full movie details for each movie
            for movie in data["results"]:
                movie_id = movie["id"]
                details_response = requests.get(
                    f"{base_url}/movie/{movie_id}",
                    params={
                        "api_key": TMDB_API_KEY,
                        "language": "en-US"
                    }
                )
                if details_response.status_code == 200:
                    movie_details = details_response.json()
                    movies.append({
                        "title": movie_details["title"],
                        "plot": movie_details["overview"],
                        "genres": ",".join([g["name"] for g in movie_details["genres"]]),
                        "poster_url": f"{TMDB_IMAGE_BASE_URL}{movie_details['poster_path']}" if movie_details['poster_path'] else None,
                        "rating": movie_details["vote_average"],
                        "year": movie_details["release_date"][:4] if movie_details["release_date"] else None,
                        "runtime": movie_details["runtime"],
                        "popularity": movie_details["popularity"]
                    })
        
        return pd.DataFrame(movies)
    
    except requests.RequestException as e:
        st.error(f"Error fetching data from TMDB: {str(e)}")
        raise

# Movie DNA Analysis
def analyze_movie_dna(title, plot, poster_url=None):
    # Theme Analysis
    themes = models['plot'](plot, 
                          candidate_labels=["action", "romance", "mystery", "comedy", "drama"])
    
    # Visual Analysis
    visual_style = None
    if poster_url:
        image = Image.open(BytesIO(requests.get(poster_url).content))
        inputs = models['processor'](images=image, return_tensors="pt")
        visual_features = models['clip'].get_image_features(**inputs)
        visual_style = visual_features.detach().numpy()
    
    # Plot Embedding
    plot_embedding = models['embedding'].encode(plot)
    
    return {
        'themes': themes,
        'embedding': plot_embedding,
        'visual_style': visual_style
    }

# Recommendation Engine
def get_recommendations(movie_id, preferences=None):
    """Enhanced recommendation engine"""
    movies_df = load_movie_data()
    base_movie = movies_df.iloc[movie_id]
    
    # Get embeddings for all movies
    all_embeddings = models['embedding'].encode(movies_df['plot'].tolist())
    base_embedding = all_embeddings[movie_id]
    
    # Calculate similarities
    plot_similarities = np.dot(all_embeddings, base_embedding)
    
    # Genre matching
    genre_boost = np.zeros(len(movies_df))
    base_genres = set(base_movie['genres'].split(','))
    for i, genres in enumerate(movies_df['genres']):
        common_genres = len(set(genres.split(',')) & base_genres)
        genre_boost[i] = common_genres * 0.1
    
    # Combine scores
    final_scores = plot_similarities + genre_boost
    
    # Get top recommendations (excluding the input movie)
    top_indices = np.argsort(final_scores)[-6:-1][::-1]
    
    return movies_df.iloc[top_indices]

# Streamlit Interface
st.title("🎬 Advanced Movie Recommendation System")

tab1, tab2, tab3 = st.tabs(["Content-Based", "Personality-Based", "Hybrid Recommendations"])

with tab1:
    st.header("Content-Based Recommendations")
    movie_input = st.text_input("Enter a movie you like:")
    
    if movie_input:
        movies_df = load_movie_data()
        
        # Find the movie in our database
        movie_match = movies_df[movies_df['title'].str.lower() == movie_input.lower()]
        
        if not movie_match.empty:
            movie = movie_match.iloc[0]
            
            # Analyze the input movie
            movie_analysis = analyze_movie_dna(
                movie['title'],
                movie['plot'],
                movie['poster_url']
            )
            
            # Show movie analysis
            st.subheader("Movie Analysis")
            st.write("Themes:")
            for label, score in zip(movie_analysis['themes']['labels'], 
                                  movie_analysis['themes']['scores']):
                st.write(f"- {label}: {score:.2%}")
            
            # Find similar movies
            all_embeddings = models['embedding'].encode(movies_df['plot'].tolist())
            similarities = np.dot(all_embeddings, movie_analysis['embedding'])
            
            # Get top 3 similar movies
            similar_indices = np.argsort(similarities)[-4:-1][::-1]
            
            st.subheader("Similar Movies")
            for idx in similar_indices:
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(movies_df.iloc[idx]['poster_url'], width=150)
                with col2:
                    st.subheader(f"{movies_df.iloc[idx]['title']} ({movies_df.iloc[idx]['year']})")
                    st.write(f"Rating: {movies_df.iloc[idx]['rating']}/10")
                    st.write(f"Genres: {movies_df.iloc[idx]['genres']}")
                    st.write(movies_df.iloc[idx]['plot'])
        else:
            st.error("Movie not found in database. Please try another movie.")

with tab2:
    st.header("Personality-Based Recommendations")
    
    # Personality quiz
    st.subheader("Quick Personality Quiz")
    mood = st.select_slider(
        "What's your current mood?",
        options=["Very Sad", "Sad", "Neutral", "Happy", "Very Happy"]
    )
    
    genre_pref = st.multiselect(
        "Select your favorite genres:",
        ["Action", "Drama", "Comedy", "Sci-Fi", "Horror", "Romance"]
    )
    
    complexity = st.slider(
        "Preferred movie complexity level",
        1, 5, 3
    )

with tab3:
    st.header("AI-Powered Hybrid Recommendations")
    
    # Free-form preference description
    user_description = st.text_area(
        "Describe your ideal movie experience:",
        "Example: I enjoy thought-provoking sci-fi movies with strong character development..."
    )
    
    if user_description:
        # Generate personalized recommendations using LLM
        prompt = f"""
        Based on the user's preferences: '{user_description}'
        Suggest 5 movies with detailed explanations why they would enjoy them.
        Consider both content similarity and emotional resonance.
        """
        
        recommendations = models['text_gen'](prompt, max_length=300, min_length=100)[0]['generated_text']
        st.write(recommendations)

# Advanced Features
st.sidebar.header("Advanced Features")
show_advanced = st.sidebar.checkbox("Show Advanced Analysis")

if show_advanced:
    st.sidebar.subheader("Movie DNA Analysis")
    # Visualization of movie characteristics 

def analyze_movie_dna(movie_text):
    # Analyze movie themes
    candidate_themes = ["action", "romance", "mystery", "comedy", "drama"]
    theme_results = models['plot'](movie_text, candidate_themes)
    
    # Analyze complexity
    complexity_labels = ["simple", "moderate", "complex"]
    complexity = models['plot'](movie_text, complexity_labels)
    
    return {
        "themes": theme_results,
        "complexity": complexity
    } 