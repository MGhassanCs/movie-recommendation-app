"""
content_recommender.py
----------------------
Implements the content-based filtering recommender using movie genres.

- Uses TF-IDF vectorization of genres (converted from pipe-delimited to space-separated)
- Computes cosine similarity matrix for all movies
- Provides function to get similar movies by genre similarity
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import streamlit as st

@st.cache_data
def build_content_recommender(movies: pd.DataFrame):
    """
    Constructs the cosine similarity matrix based on movie genres using TF-IDF vectorizer.

    Args:
        movies (pd.DataFrame): DataFrame with at least 'genres' column.

    Returns:
        numpy.ndarray: Cosine similarity matrix of shape (n_movies, n_movies).
    """
    movies = movies.copy()
    # Replace pipe delimiters in genres with spaces for vectorization
    movies['genres_str'] = movies['genres'].str.replace('|', ' ', regex=False)

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(movies['genres_str'])

    # Compute cosine similarity between all movies
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return cosine_sim

def content_recommendations(title: str, movies: pd.DataFrame, cosine_sim, top_n=5):
    """
    Given a movie title, recommend top_n most similar movies based on genre similarity.

    Args:
        title (str): Title of the reference movie.
        movies (pd.DataFrame): Movies DataFrame with 'title' column.
        cosine_sim (numpy.ndarray): Cosine similarity matrix precomputed.
        top_n (int): Number of recommendations to return (default 5).

    Returns:
        pd.DataFrame: DataFrame with recommended movies and their similarity scores.
    """
    if title not in movies['title'].values:
        raise ValueError(f"Movie '{title}' not found in dataset.")

    # Find the index of the movie that matches the title
    idx = movies[movies['title'] == title].index[0]

    # Get pairwise similarity scores for this movie with all others
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort movies by similarity score descending
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Skip the first movie (itself) and take top_n recommendations
    sim_scores = sim_scores[1:top_n+1]

    # Extract movie indices and their similarity scores
    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]

    # Prepare a DataFrame for recommended movies with similarity scores
    recs = movies.iloc[movie_indices][['movieId', 'title', 'genres']].copy()
    recs['content_score'] = scores

    return recs
