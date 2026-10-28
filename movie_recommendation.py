import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Advanced AI Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }
    .main-title {
        font-size: 2.8rem !important;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .movie-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        transition: transform 0.2s ease;
    }
    .movie-card:hover {
        transform: translateY(-3px);
        border-color: rgba(129, 140, 248, 0.4);
    }
    .badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.4);
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 5px;
        margin-top: 5px;
    }
    .score-text {
        font-weight: bold;
        color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Recommender Engine Class & Dataset
# ---------------------------------------------------------
class AdvancedMovieRecommender:
    def __init__(self, movies_data: list[dict]):
        self.df = pd.DataFrame(movies_data)
        self._prepare_data()
        self._build_similarity_matrix()

    def _prepare_data(self):
        self.df.fillna('', inplace=True)
        # Feature Soup Creation (Genres + Keywords + Director + Overview)
        self.df['content_soup'] = (
            self.df['genres'] + " " +
            self.df['keywords'] + " " +
            self.df['director'] + " " +
            self.df['overview']
        ).str.lower()

    def _build_similarity_matrix(self):
        # Advanced TF-IDF with N-grams & Sublinear TF
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['content_soup'])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def recommend(self, movie_title: str, top_n: int = 5, rating_weight: float = 0.3) -> pd.DataFrame:
        matches = self.df[self.df['title'].str.lower() == movie_title.lower()]
        if matches.empty:
            return pd.DataFrame()

        idx = matches.index[0]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        
        rec_indices = [i for i, _ in sim_scores if i != idx]
        results_df = self.df.iloc[rec_indices].copy()
        results_df['similarity_score'] = [score for i, score in sim_scores if i != idx]

        # Hybrid Scoring (Similarity + Normalized IMDb Rating)
        max_rating = self.df['rating'].max()
        results_df['normalized_rating'] = results_df['rating'] / max_rating
        
        results_df['hybrid_score'] = (
            (1 - rating_weight) * results_df['similarity_score'] + 
            rating_weight * results_df['normalized_rating']
        )

        return results_df.sort_values(by='hybrid_score', ascending=False).head(top_n)

# Modern Rich Dataset
@st.cache_resource
def load_engine():
    database = [
        {"title": "Dune: Part Two", "release_year": 2024, "rating": 8.6, "genres": "Sci-Fi Adventure Action", "director": "Denis Villeneuve", "keywords": "desert planet spice empire prophecies sandworm", "overview": "Paul Atreides unites with Chani and the Fremen while seeking revenge against conspirators."},
        {"title": "Oppenheimer", "release_year": 2023, "rating": 8.9, "genres": "Biography Drama History Science", "director": "Christopher Nolan", "keywords": "atomic bomb manhattan project world war physics quantum", "overview": "The story of J. Robert Oppenheimer and his role in the development of the atomic bomb."},
        {"title": "Interstellar", "release_year": 2014, "rating": 8.7, "genres": "Sci-Fi Space Drama Science", "director": "Christopher Nolan", "keywords": "wormhole black hole space exploration quantum gravity", "overview": "A team of explorers travel through a wormhole in space to save humanity."},
        {"title": "Inception", "release_year": 2010, "rating": 8.8, "genres": "Sci-Fi Action Thriller", "director": "Christopher Nolan", "keywords": "dreams heist subconscious mind-bending memory inception", "overview": "A thief who steals corporate secrets through dream-sharing technology is given a final chance."},
        {"title": "Blade Runner 2049", "release_year": 2017, "rating": 8.0, "genres": "Sci-Fi Mystery Cyberpunk", "director": "Denis Villeneuve", "keywords": "androids neon future detective artificial intelligence", "overview": "Young Blade Runner K's discovery of a long-buried secret leads him to track down Rick Deckard."},
        {"title": "The Dark Knight", "release_year": 2008, "rating": 9.0, "genres": "Action Crime Superhero", "director": "Christopher Nolan", "keywords": "gotham joker vigilantism corruption vigilante chaos", "overview": "When the menace known as the Joker wreaks havoc, Batman must face a psychological test."},
        {"title": "Spider-Man: Across the Spider-Verse", "release_year": 2023, "rating": 8.7, "genres": "Animation Action Multiverse Superhero", "director": "Joaquim Dos Santos", "keywords": "multiverse parallel dimensions spider-man animation heroes", "overview": "Miles Morales catapults across the Multiverse to defend its existence."},
        {"title": "Top Gun: Maverick", "release_year": 2022, "rating": 8.3, "genres": "Action Drama Aviation", "director": "Joseph Kosinski", "keywords": "fighter jet pilot navy speed military mission instructor", "overview": "After thirty years, Maverick trains a detachment of graduates for a specialized mission."}
    ]
    return AdvancedMovieRecommender(database)

recommender = load_engine()

# ---------------------------------------------------------
# 3. Streamlit Interface & Layout
# ---------------------------------------------------------
st.markdown('<div class="main-title">🎬 CineMind AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced Hybrid Recommendation Engine (TF-IDF + Cosine Similarity + IMDb Weights)</div>', unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Model Parameters")
    rating_weight = st.slider("⚖️ IMDb Rating Weight", 0.0, 1.0, 0.3, 0.05, 
                              help="Higher weight prioritizes higher rated movies alongside content similarity.")
    top_n = st.slider("🔢 Number of Recommendations", 1, 5, 3)
    
    st.markdown("---")
    st.markdown("### 🧠 Backend Specs")
    st.info("• **Vectorization:** TF-IDF (N-grams 1-2)\n• **Scoring:** Hybrid Vector Dot-Product\n• **Features:** Soup (Genres, Director, Keywords, Plot)")

# Main Selection
col1, col2 = st.columns([3, 1])
with col1:
    selected_movie = st.selectbox("🔍 Select a movie you like:", recommender.df['title'].tolist())

# Selected Movie Profile Card
selected_row = recommender.df[recommender.df['title'] == selected_movie].iloc[0]
st.markdown("### 📌 Selected Movie Profile")
st.markdown(f"""
<div class="movie-card" style="border-left: 5px solid #38bdf8;">
    <h3>{selected_row['title']} <span style="color:#94a3b8; font-size:1rem;">({selected_row['release_year']})</span></h3>
    <p>⭐ <b>Rating:</b> {selected_row['rating']}/10 | 🎬 <b>Director:</b> {selected_row['director']}</p>
    <p>{selected_row['overview']}</p>
    <div>
        {' '.join([f'<span class="badge">{g}</span>' for g in selected_row['genres'].split()])}
    </div>
</div>
""", unsafe_allow_html=True)

# Recommendation Trigger
if st.button("🚀 Generate Recommendations", use_container_width=True):
    recommendations = recommender.recommend(selected_movie, top_n=top_n, rating_weight=rating_weight)
    
    st.markdown("---")
    st.subheader(f"✨ Top Recommendations for '{selected_movie}':")
    
    cols = st.columns(len(recommendations))
    for i, (_, row) in enumerate(recommendations.iterrows()):
        with cols[i]:
            st.markdown(f"""
            <div class="movie-card">
                <h4>🍿 {row['title']}</h4>
                <p style="font-size:0.85rem; color:#94a3b8;">({row['release_year']}) | ⭐ {row['rating']}/10</p>
                <p style="font-size:0.85rem;">{row['overview'][:90]}...</p>
                <hr style="border-color: rgba(255,255,255,0.1);">
                <p style="font-size:0.8rem; margin:0;"><b>Similarity:</b> <span class="score-text">{row['similarity_score']*100:.1f}%</span></p>
                <p style="font-size:0.8rem; margin:0;"><b>Hybrid Score:</b> <span class="score-text">{row['hybrid_score']:.3f}</span></p>
                <div>
                    {' '.join([f'<span class="badge">{g}</span>' for g in row['genres'].split()])}
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.caption("💡 *Built using Streamlit, Scikit-Learn TF-IDF, and Python Object-Oriented Architecture.*")