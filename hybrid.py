"""
hybrid.py
---------
Combines content-based and collaborative filtering recommendations
into a hybrid recommender system.

- Normalizes scores from both recommenders
- Adds normalized scores together
- Returns the top combined recommendations
"""

import pandas as pd
from content_recommender import content_recommendations
from collaborative import collaborative_recommendations

def hybrid_recommendations(user_id, fav_movie, movies, ratings, cosine_sim, algo, top_n=5):
    """
    Generate hybrid recommendations combining collaborative and content filtering.

    Args:
        user_id (int): MovieLens user ID.
        fav_movie (str): User's favorite movie title.
        movies (pd.DataFrame): Movies DataFrame.
        ratings (pd.DataFrame): Ratings DataFrame.
        cosine_sim: Cosine similarity matrix from content recommender.
        algo: Trained collaborative filtering model.
        top_n (int): Number of recommendations to return (default 5).

    Returns:
        pd.DataFrame: Top recommended movies with titles and genres.
    """
    # Get collaborative filtering recommendations
    try:
        collab_recs = collaborative_recommendations(user_id, movies, ratings, algo, top_n=top_n*2)
    except ValueError:
        collab_recs = pd.DataFrame(columns=['movieId', 'title', 'genres', 'collab_score'])

    # Get content-based filtering recommendations
    try:
        content_recs = content_recommendations(fav_movie, movies, cosine_sim, top_n=top_n*2)
    except ValueError:
        content_recs = pd.DataFrame(columns=['movieId', 'title', 'genres', 'content_score'])

    # If both have recommendations, merge and combine scores
    if not collab_recs.empty and not content_recs.empty:
        merged = pd.merge(collab_recs, content_recs, on=['movieId', 'title', 'genres'], how='outer')

        # Fill missing scores with zero
        merged['collab_score'] = merged['collab_score'].fillna(0)
        merged['content_score'] = merged['content_score'].fillna(0)

        # Normalize scores to range [0,1]
        if merged['collab_score'].max() > 0:
            merged['collab_score'] /= merged['collab_score'].max()
        if merged['content_score'].max() > 0:
            merged['content_score'] /= merged['content_score'].max()

        # Combine normalized scores
        merged['hybrid_score'] = merged['collab_score'] + merged['content_score']

        # Sort by combined score descending
        merged = merged.sort_values('hybrid_score', ascending=False)

        recs = merged.head(top_n)
    else:
        # If one recommender is empty, fallback to the other
        recs = pd.concat([collab_recs, content_recs]).drop_duplicates(subset=['movieId']).head(top_n)

    return recs[['title', 'genres']]
