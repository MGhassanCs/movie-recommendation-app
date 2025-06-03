import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import streamlit as st

# === 1. Load Data ===
@st.cache_data
def load_data():
    movies_path = 'ml-1m/movies.dat'
    ratings_path = 'ml-1m/ratings.dat'

    movies = pd.read_csv(
        movies_path, 
        sep='::', 
        engine='python', 
        names=['movieId', 'title', 'genres'], 
        encoding='latin-1'
    )
    movies['movieId'] = movies['movieId'].astype(int)

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

# === 2. Content-based recommender ===
@st.cache_data
def build_content_recommender(movies):
    movies = movies.copy()
    movies['genres_str'] = movies['genres'].str.replace('|', ' ', regex=False)
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(movies['genres_str'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def content_recommendations(title, movies, cosine_sim, top_n=5):
    if title not in movies['title'].values:
        raise ValueError(f"Movie '{title}' not found in dataset.")
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # exclude itself
    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    recs = movies.iloc[movie_indices][['movieId', 'title', 'genres']].copy()
    recs['content_score'] = scores
    return recs

# === 3. Collaborative filtering recommender ===
@st.cache_resource
def build_collaborative_model(ratings):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, _ = train_test_split(data, test_size=0.25, random_state=42)
    algo = SVD()
    algo.fit(trainset)
    return algo

def collaborative_recommendations(user_id, movies, ratings, algo, top_n=5):
    if user_id not in ratings['userId'].unique():
        raise ValueError(f"User ID {user_id} not found in dataset.")
    user_rated = set(ratings[ratings.userId == user_id]['movieId'])
    movie_ids = movies['movieId'].unique()
    predictions = [algo.predict(user_id, int(mid)) for mid in movie_ids if mid not in user_rated]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_preds = predictions[:top_n]
    top_movie_ids = [int(pred.iid) for pred in top_preds]
    scores = [pred.est for pred in top_preds]
    recs = movies[movies['movieId'].isin(top_movie_ids)][['movieId', 'title', 'genres']].copy()
    recs['collab_score'] = scores
    return recs

# === 4. Hybrid recommender ===
def hybrid_recommendations(user_id, fav_movie, movies, ratings, cosine_sim, algo, top_n=5):
    try:
        collab_recs = collaborative_recommendations(user_id, movies, ratings, algo, top_n=top_n*2)
    except ValueError:
        collab_recs = pd.DataFrame(columns=['movieId', 'title', 'genres', 'collab_score'])

    try:
        content_recs = content_recommendations(fav_movie, movies, cosine_sim, top_n=top_n*2)
    except ValueError:
        content_recs = pd.DataFrame(columns=['movieId', 'title', 'genres', 'content_score'])

    # Merge on movieId, combine scores (normalize first)
    if not collab_recs.empty and not content_recs.empty:
        merged = pd.merge(collab_recs, content_recs, on=['movieId', 'title', 'genres'], how='outer')
        # Fill NaN with zeros
        merged['collab_score'] = merged['collab_score'].fillna(0)
        merged['content_score'] = merged['content_score'].fillna(0)
        # Normalize scores
        if merged['collab_score'].max() > 0:
            merged['collab_score'] = merged['collab_score'] / merged['collab_score'].max()
        if merged['content_score'].max() > 0:
            merged['content_score'] = merged['content_score'] / merged['content_score'].max()
        merged['hybrid_score'] = merged['collab_score'] + merged['content_score']
        merged = merged.sort_values('hybrid_score', ascending=False)
        recs = merged.head(top_n)
    else:
        recs = pd.concat([collab_recs, content_recs]).drop_duplicates(subset=['movieId']).head(top_n)
    return recs[['title', 'genres']]

# === 5. Streamlit App UI ===
def main():
    st.set_page_config(page_title="Hybrid Movie Recommender", layout="centered")
    st.title("🎬 Hybrid Movie Recommender System")

    st.write("""
    Welcome! This app recommends movies by combining **collaborative filtering** (user-based ratings) and **content-based filtering** (movie genres).
    *Enter your MovieLens User ID and select a movie you like to receive personalized recommendations!*
    """)

    movies, ratings = load_data()
    cosine_sim = build_content_recommender(movies)
    algo = build_collaborative_model(ratings)

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
                    for idx, row in recs.iterrows():
                        st.markdown(f"**{row['title']}**  \nGenres: _{row['genres']}_")
            except Exception as e:
                st.error(f"Error: {e}")

    st.info("Tip: The more you've rated in MovieLens, the better your recommendations!")

if __name__ == "__main__":
    main()
