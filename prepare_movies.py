# prepare_movies.py
import pandas as pd
from collections import Counter

def prepare_movie_dataset(input_csv="movies.csv", output_csv="movies_cleaned_for_knn.csv"):
    """
    Loads the movie CSV, handles missing values, creates a 'content' field for ML,
    and saves a cleaned CSV suitable for KNN-based content recommender.
    
    Parameters:
    - input_csv: str, path to the original movie CSV
    - output_csv: str, path to save the cleaned CSV
    """
    # Load dataset
    movies = pd.read_csv(input_csv)
    print(f"Loaded {len(movies)} movies.")

    # Check for missing values
    missing = movies.isnull().sum()
    print("Missing values per column:\n", missing)

    # Fill missing values if needed
    for col in ['genre','name','director_name', 'certificate']:
        if col in movies.columns:
            movies[col] = movies[col].fillna('')

    # Create 'content' field for TF-IDF
    # Combine title, genre, certificate, director
    movies['content'] = (
        movies['name'].astype(str) + ' ' +
        movies['genre'].astype(str) + ' ' +
        movies['certificate'].astype(str) + ' ' +
        movies['director_name'].astype(str)
    )

    # Preview
    print("Sample content field:")
    print(movies[['name','content']].head(5))

    # Save cleaned CSV
    movies.to_csv(output_csv, index=False)
    print(f"Cleaned dataset saved to {output_csv}")

    return movies

if __name__ == "__main__":
    prepare_movie_dataset()
