import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib

def train_knn_recommender(cleaned_csv="movies_cleaned_for_knn.csv", artifact_dir="artifacts", top_n_neighbors=10):
    """
    Trains a content-based KNN movie recommender using TF-IDF on the 'content' field.
    
    Parameters:
    - cleaned_csv: path to the cleaned movie CSV
    - artifact_dir: folder to save artifacts (TF-IDF and KNN model)
    - top_n_neighbors: number of neighbors to consider in the model (default 10)
    """
    import os
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Load cleaned CSV
    movies = pd.read_csv(cleaned_csv)
    print(f"Loaded {len(movies)} movies for training.")

    # Check that 'content' exists
    if 'content' not in movies.columns:
        raise ValueError("The CSV must have a 'content' column for TF-IDF.")

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(ngram_range=(1,2), stop_words='english', min_df=1)
    tfidf_matrix = tfidf.fit_transform(movies['content'].astype(str))
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Train NearestNeighbors model (cosine similarity)
    nn_model = NearestNeighbors(n_neighbors=top_n_neighbors+1, metric='cosine', algorithm='brute')
    nn_model.fit(tfidf_matrix)
    print("KNN model trained.")

    # Save artifacts
    joblib.dump(tfidf, f"{artifact_dir}/tfidf_vectorizer.joblib")
    joblib.dump(tfidf_matrix, f"{artifact_dir}/tfidf_matrix.joblib")
    joblib.dump(nn_model, f"{artifact_dir}/nearest_neighbors.joblib")
    movies.to_pickle(f"{artifact_dir}/movies_df.pkl")
    print(f"Artifacts saved in '{artifact_dir}' folder.")

    return movies, tfidf, tfidf_matrix, nn_model

if __name__ == "__main__":
    movies, tfidf, tfidf_matrix, nn_model = train_knn_recommender()