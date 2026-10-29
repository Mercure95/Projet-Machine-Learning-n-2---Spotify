import streamlit as st

# 1. Configuration de la page (Logo dans l'onglet du navigateur)
LOGO_ICON = "https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg"
LOGO_FULL = "https://upload.wikimedia.org/wikipedia/commons/2/26/Spotify_logo_with_text.svg"

st.set_page_config(page_title="Spotify ML - Accueil", page_icon=LOGO_ICON, layout="wide")

# 2. Logo dans le menu latéral (Streamlit >= 1.31)
st.logo(LOGO_FULL)

# 3. CSS Global "DA Spotify" (Noir et Vert)
st.markdown("""
    <style>
    /* Masquer la barre supérieure par défaut de Streamlit */
    [data-testid="stHeader"] {
        display: none;
    }
    
    /* Fond principal noir Spotify */
    .stApp {
        background-color: #121212;
        color: #FFFFFF;
    }
    
    /* Barre latérale noire très foncée */
    [data-testid="stSidebar"] {
        background-color: #000000;
    }
    
    /* Forcer tous les titres en Vert Spotify avec la bonne police */
    h1, h2, h3 {
        color: #1DB954 !important;
        font-weight: 900 !important;
        letter-spacing: -1px;
    }
    
    /* Styliser les métriques */
    div[data-testid="stMetricValue"] {
        color: #1DB954;
    }
    
    /* Barre de progression */
    .stProgress > div > div > div > div {
        background-color: #1DB954;
    }
    </style>
""", unsafe_allow_html=True)

# 4. Titre de la page avec Logo intégré
st.markdown(f"""
<div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
    <img src="{LOGO_ICON}" width="65">
    <h1 style="margin: 0; font-size: 55px;">Projet Machine Learning</h1>
</div>
""", unsafe_allow_html=True)

st.write("Bienvenue sur l'outil d'analyse musicale développé pour décrypter le catalogue Spotify.")

st.info("👈 **Naviguez dans le menu de gauche** pour explorer les données, tester l'algorithme de Régression (énergie) ou l'algorithme de Classification (genres musicaux).")

st.header("Le jeu de données")
st.markdown("""
* **Source :** TidyTuesday (API Spotify)
* **Volume :** ~32 800 morceaux
* **Objectif :** Décrypter la musique à travers ses composantes mathématiques (tempo, volume, acoustique, etc.).
""")