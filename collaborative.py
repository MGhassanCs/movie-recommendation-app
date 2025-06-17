"""
collaborative.py
----------------
Implements collaborative filtering based recommender system using
the Surprise library's SVD algorithm.

- Loads rating data into Surprise dataset format
- Trains an SVD model on the train split
- Provides function to generate personalized recommendations
"""

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pandas as pd
import streamlit as st

@st.cache_resource
def build_collaborative_model(ratings: pd.DataFrame):
    """
    Builds and trains the SVD collaborative filtering model.

    Args:
        ratings (pd.DataFrame): DataFrame with columns ['userId', 'movieId', 'rating'].

    Returns:
        surprise.prediction_algorithms.matrix_factorization.SVD: Trained SVD model.
    """
    reader = Reader(rating_scale=(1, 5))

    # Load ratings into Surprise dataset format
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    # Split into training and test set (though test unused here)
    trainset, _ = train_test_split(data, test_size=0.25, random_state=42)

    # Initialize and train SVD algorithm
    algo = SVD()
    algo.fit(trainset)

    return algo

def collaborative_recommendations(user_id: int, movies: pd.DataFrame, ratings: pd.DataFrame, algo, top_n=5):
    """
    Generate top_n movie recommendations for a user based on collaborative filtering.

    Args:
        user_id (int): MovieLens user ID.
        movies (pd.DataFrame): Movies DataFrame.
        ratings (pd.DataFrame): Ratings DataFrame.
        algo: Trained Surprise SVD model.
        top_n (int): Number of recommendations to return (default 5).

    Returns:
        pd.DataFrame: DataFrame of recommended movies with predicted scores.
    """
    if user_id not in ratings['userId'].unique():
        raise ValueError(f"User ID {user_id} not found in dataset.")

    # Get movies already rated by the user to exclude them
    user_rated = set(ratings[ratings.userId == user_id]['movieId'])

    movie_ids = movies['movieId'].unique()

    # Predict ratings for all movies not yet rated by user
    predictions = [algo.predict(user_id, int(mid)) for mid in movie_ids if mid not in user_rated]

    # Sort predictions by estimated rating descending
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Take top_n predictions
    top_preds = predictions[:top_n]

    top_movie_ids = [int(pred.iid) for pred in top_preds]
    scores = [pred.est for pred in top_preds]

    # Prepare DataFrame of recommended movies with predicted scores
    recs = movies[movies['movieId'].isin(top_movie_ids)][['movieId', 'title', 'genres']].copy()
    recs['collab_score'] = scores

    return recs
