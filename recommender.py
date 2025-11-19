import pandas as pd
import joblib

ARTIFACT_DIR = "artifacts"

def load_recommender():
    """
    Loads the trained TF-IDF vectorizer, TF-IDF matrix, NearestNeighbors model,
    and movies dataframe. Returns them as a tuple.
    """
    tfidf = joblib.load(f"{ARTIFACT_DIR}/tfidf_vectorizer.joblib")
    tfidf_matrix = joblib.load(f"{ARTIFACT_DIR}/tfidf_matrix.joblib")
    nn_model = joblib.load(f"{ARTIFACT_DIR}/nearest_neighbors.joblib")
    movies_df = pd.read_pickle(f"{ARTIFACT_DIR}/movies_df.pkl")
    return tfidf, tfidf_matrix, nn_model, movies_df


def get_recommendations(movie_title, tfidf, tfidf_matrix, nn_model, movies_df, top_n=10):
    """
    Returns top-N similar movies based on content.
    
    Parameters:
    - movie_title: string, title of the movie to query
    - tfidf: trained TF-IDF vectorizer
    - tfidf_matrix: TF-IDF matrix of all movies
    - nn_model: trained NearestNeighbors model
    - movies_df: pandas DataFrame with movie metadata
    - top_n: number of recommendations to return
    
    Returns:
    - List of dictionaries with keys: name, genre, imdb_rating, imbd_votes, score
    """
    # Find movie index
    matches = movies_df[movies_df['name'].str.contains(movie_title, case=False, na=False)]
    if matches.empty:
        return []

    idx = matches.index[0]
    movie_vec = tfidf_matrix[idx]

    # Query nearest neighbors
    distances, indices = nn_model.kneighbors(movie_vec, n_neighbors=top_n+1)
    results = []
    for dist, i in zip(distances.flatten(), indices.flatten()):
        if i == idx:
            continue  # skip the movie itself
        row = movies_df.iloc[i]
        results.append({
            'name': row['name'],
            'genre': row['genre'],
            'imdb_rating': row['imdb_rating'],
            'imbd_votes': row['imbd_votes'],
            'score': 1 - dist  # similarity score
        })
        if len(results) >= top_n:
            break
    return results