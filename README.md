# Movie Recommender

A **content-based movie recommender** built with **Streamlit** and **scikit-learn**.  
It uses a **TF-IDF vectorizer** and **KNN model** on movie metadata to suggest similar movies. Posters, genres, and IMDb ratings are displayed in a Netflix-inspired UI.

---

## What it Uses

- **Python 3.11+**
- [Streamlit](https://streamlit.io/)
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [joblib](https://joblib.readthedocs.io/)
- Preprocessed movie dataset (`movies_cleaned_for_knn.csv`)
- Trained artifacts (TF-IDF vectorizer, TF-IDF matrix, KNN model, movie DataFrame)
