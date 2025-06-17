"""
data_loader.py
---------------
Module responsible for loading and caching MovieLens dataset files:
- movies.dat
- ratings.dat

Uses Streamlit's cache decorator to improve performance by avoiding
reloading data on every app rerun.
"""

import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    """
    Loads MovieLens 1M movies and ratings datasets from local files.

    Returns:
        tuple: (movies DataFrame, ratings DataFrame)
    """
    movies_path = 'ml-1m/movies.dat'
    ratings_path = 'ml-1m/ratings.dat'

    # Load movies data with proper parsing and encoding
    movies = pd.read_csv(
        movies_path,
        sep='::',
        engine='python',
        names=['movieId', 'title', 'genres'],
        encoding='latin-1'
    )
    movies['movieId'] = movies['movieId'].astype(int)

    # Load ratings data with proper parsing and encoding
    ratings = pd.read_csv(
        ratings_path,
        sep='::',
        engine='python',
        names=['userId', 'movieId', 'rating', 'timestamp'],
        encoding='latin-1'
    )
    ratings['userId'] = ratings['userId'].astype(int)
    ratings['movieId'] = ratings['movieId'].astype(int)

    return movies, ratings
