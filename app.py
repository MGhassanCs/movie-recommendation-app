"""
app.py
-------
Main Streamlit application entry point for the Hybrid Movie Recommender System.

- Loads data and builds recommenders
- Provides user input for User ID and favorite movie
- Displays recommendations in a user-friendly UI
"""

import streamlit as st
from data_loader import load_data
from content_recommender import build_content_recommender
from collaborative import build_collaborative_model
from hybrid import hybrid_recommendations

def main():
    """
    Runs the Streamlit app for hybrid movie recommendations.
    """
    st.set_page_config(page_title="Hybrid Movie Recommender", layout="centered")
    st.title("🎬 Hybrid Movie Recommender System")

    st.write("""
    Welcome! This app recommends movies by combining **collaborative filtering** (user-based ratings) and **content-based filtering** (movie genres).
    *Enter your MovieLens User ID and select a movie you like to receive personalized recommendations!*
    """)

    # Load data and models
    movies, ratings = load_data()
    cosine_sim = build_content_recommender(movies)
    algo = build_collaborative_model(ratings)

    # User inputs
    user_id = st.number_input(
        "Enter your MovieLens User ID",
        min_value=int(ratings['userId'].min()),
        max_value=int(ratings['userId'].max()),
        step=1,
        value=int(ratings['userId'].min()),
        help="User IDs are numeric and can be found in the ratings file."
    )

    all_titles = movies['title'].sort_values().unique()
    fav_movie = st.selectbox(
        "Choose a favorite movie (for content-based recommendations)",
        all_titles,
        index=0
    )

    if st.button("Get Recommendations"):
        with st.spinner("Finding the best movies for you..."):
            try:
                recs = hybrid_recommendations(user_id, fav_movie, movies, ratings, cosine_sim, algo)
                if recs.empty:
                    st.warning("No recommendations found. Try a different user ID or movie.")
                else:
                    st.success("Here are some recommendations for you:")
                    for _, row in recs.iterrows():
                        st.markdown(f"**{row['title']}**  \nGenres: _{row['genres']}_")
            except Exception as e:
                st.error(f"Error: {e}")

    st.info("Tip: The more you've rated in MovieLens, the better your recommendations!")

if __name__ == "__main__":
    main()
