import streamlit as st
from recommender import load_recommender, get_recommendations

# Load trained artifacts
tfidf, tfidf_matrix, nn_model, movies_df = load_recommender()

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