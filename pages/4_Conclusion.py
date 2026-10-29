import streamlit as st

# =========================================================
# --- BLOC DA SPOTIFY À METTRE AU DÉBUT DE CHAQUE PAGE ---
# =========================================================
LOGO_ICON = "https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg"
LOGO_FULL = "https://upload.wikimedia.org/wikipedia/commons/2/26/Spotify_logo_with_text.svg"

# Logo dans l'onglet du navigateur
st.set_page_config(page_title="Bilan et Conclusion", page_icon=LOGO_ICON, layout="wide")

# Logo dans la barre latérale
st.logo(LOGO_FULL)

# Thème CSS Noir/Vert pour la page
st.markdown("""
    <style>
    .stApp { background-color: #121212; color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #000000; }
    h1, h2, h3 { color: #1DB954 !important; font-weight: 900 !important; letter-spacing: -1px; }
    div[data-testid="stMetricValue"] { color: #1DB954; }
    .stProgress > div > div > div > div { background-color: #1DB954; }
    .big-font {
        font-size: 45px !important;
        font-weight: 900;
        color: #1DB954;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 20px;
        text-shadow: 0px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: -1px;
    }
    [data-testid="stHeader"] {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

# Titre avec le logo Spotify à côté
st.markdown(f"""
<div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
    <img src="{LOGO_ICON}" width="50">
    <h1 style="margin: 0; font-size: 40px;">Bilan et Conclusion</h1>
</div>
""", unsafe_allow_html=True)
# ---------------------------------------------------------

st.markdown("Cette page synthétise les apprentissages de notre projet et apporte une réponse définitive à la start-up musicale concernant l'analyse technique des morceaux.")
st.divider()

# --- 1. RÉPONSES AUX PROBLÉMATIQUES ---
st.header("1. Réponses aux problématiques")

col_r1, col_r2, col_r3 = st.columns(3)

with col_r1:
    st.success("⚡ **Énergie (Succès)**")
    st.markdown("""
    **Oui, il est tout à fait possible de prédire l'énergie d'un morceau avec précision.**
    * Le modèle Multiple (Random Forest) a prouvé son efficacité.
    * **Moteurs principaux :** Le volume (`loudness`) est le principal vecteur de dynamisme, tandis que l'`acousticness` agit comme un frein naturel.
    * **Conclusion :** L'énergie est une métrique purement physique et acoustique, facilement compréhensible par une IA.
    """)

with col_r2:
    st.warning("🔥 **Popularité (Échec Instructif)**")
    st.markdown("""
    **Non, les données audio ne suffisent pas à prédire le succès.**
    * Nos modèles ont échoué à deviner la popularité d'un titre à partir de son BPM ou de sa "danceability".
    * **Conclusion sociologique :** La popularité n'est pas mathématique. Elle dépend de la notoriété de l'artiste, du marketing, et du contexte temporel (ex: une tendance TikTok), des données absentes de l'audio brut.
    """)

with col_r3:
    st.info("🎵 **Genre Musical (Plafond de verre)**")
    st.markdown("""
    **La prédiction est possible, mais bloquée autour de 60% d'exactitude.**
    * **Reconnaissance facile :** L'EDM et le Rock ont des signatures acoustiques uniques (rythmes, saturation).
    * **Confusion naturelle :** La Pop est un genre "carrefour" qui emprunte au R&B, au Rap et au Latin.
    * **Conclusion :** Ce chevauchement prouve que la classification stricte a ses limites : la musique moderne est hybride.
    """)

st.divider()

# --- 2. LIMITES DES MODÈLES ---
st.header("2. Limites identifiées")
col_l1, col_l2 = st.columns(2)

with col_l1:
    st.markdown("### 📉 Limites Techniques")
    st.markdown("""
    * **Subjectivité des Playlists :** L'assignation d'un morceau à un genre dépend d'un humain (ou d'un curateur Spotify). Un même morceau peut objectivement appartenir à une playlist Pop et à une playlist Dance, faussant l'entraînement de l'IA.
    * **Les "Boîtes Noires" Spotify :** Les indicateurs comme la *danceability* ou la *valence* sont des variables déjà pré-calculées par l'algorithme propriétaire de Spotify. Nous faisons confiance à une IA pour entraîner notre propre IA.
    """)

with col_l2:
    st.markdown("### 🎼 Limites Musicales")
    st.markdown("""
    * **Absence du texte (Lyrics) :** Différencier le Rap du R&B repose énormément sur le débit de paroles et le vocabulaire, une dimension totalement absente de nos données actuelles.
    * **Absence du contexte artiste :** Ne pas intégrer la taille de la "fanbase" d'un artiste rend la prédiction de la popularité impossible.
    """)

st.divider()

# --- 3. PISTES D'AMÉLIORATION ---
st.header("3. Pistes d'amélioration")
st.markdown("""
Pour aller plus loin et fournir un outil véritablement robuste (notamment pour détecter la popularité et affiner les genres), la start-up devrait envisager :

1. **Intégrer le NLP (Traitement du Langage Naturel) :** Scraper et analyser les paroles des musiques pour aider la classification (crucial pour le Rap et la Pop).
2. **Scraper les métadonnées sociales :** Ajouter le nombre d'abonnés de l'artiste ou le nombre d'apparitions dans les playlists éditoriales pour enfin réussir à prédire la variable `track_popularity`.
3. **Traitement du signal audio brut :** Utiliser des réseaux de neurones profonds (Deep Learning) directement sur les spectrogrammes des fichiers MP3 (les fréquences réelles), pour s'affranchir des "boîtes noires" de l'API Spotify.
""")

st.divider()
st.markdown("<h3 style='text-align: center; color: #1DB954;'> Merci pour votre attention ! L'équipe est à votre disposition pour vos questions.</h3>", unsafe_allow_html=True)