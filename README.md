# Hybrid Movie Recommender System

This project is a **hybrid movie recommender system** that suggests movies based on both your personal ratings and the content of movies (genres). It uses the [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/) and combines collaborative filtering (user-based) with content-based filtering (genre similarity). The app is built using Python, pandas, scikit-learn, Surprise, and Streamlit.

---

## Features

- **Content-based filtering:** Recommends movies similar in genres to your chosen favorite.
- **Collaborative filtering:** Suggests movies based on what similar users liked.
- **Hybrid recommendations:** Blends both methods for personalized, relevant suggestions.
- **Interactive web app:** Easy-to-use Streamlit UI for exploring recommendations.

---

## How It Works

1. **Data Loading:** Reads `movies.dat` and `ratings.dat` from the MovieLens 1M dataset.
2. **Content-Based Filtering:** Uses TF-IDF vectorization on genres and cosine similarity to find similar movies.
3. **Collaborative Filtering:** Uses the Surprise SVD model to predict ratings for unseen movies for a given user.
4. **Hybrid Model:** Merges both approaches, normalizes scores, and combines them for final recommendations.
5. **Web App:** Input your MovieLens user ID and pick a favorite movie to receive your recommendations.

---

## Installation

1. **Clone this repository:**
    ```bash
    git clone https://github.com/yourusername/hybrid-movie-recommender.git
    cd hybrid-movie-recommender
    ```

2. **Download MovieLens 1M Dataset:**
    - Download from [here](https://grouplens.org/datasets/movielens/1m/)
    - Extract the files so you have the `ml-1m` folder in your project directory.

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    Example `requirements.txt`:
    ```
    streamlit
    pandas
    scikit-learn
    scikit-surprise
    ```

---

## Usage

1. **Run the app:**
    ```bash
    streamlit run app.py
    ```

2. **Web interface:**
    - Enter your MovieLens User ID (from `ratings.dat`).
    - Choose a favorite movie from the dropdown.
    - Click "Get Recommendations" to see your personalized list.

---

## File Structure

- `app.py` : Main Streamlit application file.
- `ml-1m/` : Folder containing MovieLens 1M data files (`movies.dat`, `ratings.dat`, etc.).
- `requirements.txt` : List of required Python packages.
- `README.md` : This documentation file.

---

## Notes

- **User ID:** Must be an existing user in the MovieLens dataset.
- **Performance:** First run may take some time as models are built and cached.
- **Customization:** You can adjust the number of recommendations or blend method in `app.py`.

---

## License

This project is for educational purposes. Dataset provided by [GroupLens](https://grouplens.org/datasets/movielens/).

---

## Acknowledgments

- [MovieLens Datasets](https://grouplens.org/datasets/movielens/)
- [Surprise Recommender Library](https://surpriselib.com/)
- [Streamlit](https://streamlit.io/)

# intermediate-project
