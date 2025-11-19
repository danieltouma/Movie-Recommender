import streamlit as st
from recommender import load_recommender, get_recommendations

# Load trained artifacts
tfidf, tfidf_matrix, nn_model, movies_df = load_recommender()

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    /* Dark background but readable */
    .stApp {
        background-color: #1c1c1c;
        color: #fff;
        font-family: 'Helvetica', sans-serif;
    }

    /* App title */
    .stTitle {
        color: #e50914;
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 2rem;
    }

    /* Grid layout: 3 movies per row */
    .recommendations-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        padding: 10px 0;
    }

    /* Movie card */
    .movie-card {
        position: relative;
        border-radius: 10px;
        overflow: hidden;
        cursor: pointer;
        transition: transform 0.3s, box-shadow 0.3s;
        background-color: #2a2a2a;
    }

    .movie-card:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    }

    /* Movie poster fills card */
    .movie-card img {
        width: 100%;
        height: auto;
        display: block;
    }

    /* Overlay for text */
    .overlay {
        position: absolute;
        bottom: 0;
        width: 100%;
        background: linear-gradient(transparent, rgba(0,0,0,0.8));
        color: #fff;
        padding: 10px;
        box-sizing: border-box;
    }

    .movie-title {
        font-weight: bold;
        font-size: 1.1em;
        margin: 0;
    }

    .movie-details {
        font-size: 0.9em;
        color: #ddd;
        margin: 0;
    }

    select, button {
        background-color: #e50914;
        color: #fff;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
        margin-top: 1rem;
    }

    select {
        margin-bottom: 1rem;
    }

    button:hover {
        background-color: #f6121d;
    }

    /* Subheader styling */
    .stSubheader {
        color: #fff;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="stTitle">🎬 Movie Recommender</h1>', unsafe_allow_html=True)

# Movie selection
movie_query = st.selectbox(
    "Select a movie:",
    options=movies_df['name'].tolist()
)

# Recommendation button
if st.button("Recommend") and movie_query:
    results = get_recommendations(movie_query, tfidf, tfidf_matrix, nn_model, movies_df)
    if results:
        st.subheader(f"Top recommendations for '{movie_query}':")
        st.markdown('<div class="recommendations-grid">', unsafe_allow_html=True)
        for r in results:
            img_url = movies_df[movies_df['name'] == r['name']]['img_link'].values
            poster = img_url[0] if len(img_url) > 0 else ''
            st.markdown(f'''
                <div class="movie-card">
                    <img src="{poster}" alt="{r["name"]}">
                    <div class="overlay">
                        <div class="movie-title">{r["name"]}</div>
                        <div class="movie-details">{r["genre"]} | ⭐ {r["imdb_rating"]}</div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.write("No recommendations found.")