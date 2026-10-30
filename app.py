import streamlit as st
import pandas as pd

# Import de la fonction de chargement des données depuis data.py
# (Assure-toi que app.py et data.py sont bien dans le même dossier racine)
from data import load_data

# =========================================================
# Configuration de la page et DA Spotify
# =========================================================
LOGO_ICON = "https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg"
LOGO_FULL = "https://upload.wikimedia.org/wikipedia/commons/2/26/Spotify_logo_with_text.svg"

st.set_page_config(page_title="Spotify ML - Accueil", page_icon=LOGO_ICON, layout="wide")

# Logo dans le menu latéral
st.logo(LOGO_FULL)

# CSS Global "DA Spotify" (Noir et Vert)
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
    
    /* Barre latérale noire */
    [data-testid="stSidebar"] {
        background-color: #000000;
    }
    
    /* Forcer tous les titres en Vert Spotify avec la bonne police */
    h1, h2, h3 {
        color: #1DB954 !important;
        font-weight: 900 !important;
        letter-spacing: -1px;
    }
    
    /* Styliser les tableaux pour qu'ils s'intègrent au thème sombre */
    [data-testid="stTable"] {
        background-color: #1C1C1C;
        border-radius: 8px;
    }
    th {
        color: #1DB954 !important;
    }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# En-tête
# =========================================================
st.markdown(f"""
<div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
    <img src="{LOGO_ICON}" width="65">
    <h1 style="margin: 0; font-size: 55px;">Projet Machine Learning</h1>
</div>
""", unsafe_allow_html=True)

st.write("Bienvenue sur l'outil d'analyse musicale développé pour décrypter le catalogue Spotify.")
st.markdown("**Équipe du projet :** *MARILLEAU Bastien, MARCHAND Erwann, GOASDUFF Pierre, SOICHET Quentin*")

st.info("👈 **Naviguez dans le menu de gauche** pour explorer les données, tester l'algorithme de Régression (énergie) ou l'algorithme de Classification (genres musicaux).")

st.divider()

# =========================================================
# 1. Présentation et Aperçu du jeu de données
# =========================================================
st.header("1. Présentation du catalogue musical")
st.markdown("""
* **Source :** TidyTuesday (Données issues de l'API officielle Spotify)
* **Volume brut :** ~32 800 morceaux de musique
* **Objectif :** Analyser et décrypter la musique à travers ses composantes mathématiques et techniques (tempo, volume, acoustique, etc.).
""")

with st.spinner("Chargement des données depuis la base..."):
    try:
        df = load_data()
        st.subheader("Aperçu brut (50 premiers morceaux)")
        # On affiche seulement 50 lignes pour ne pas surcharger le navigateur Web
        st.dataframe(df.head(50), use_container_width=True)
    except Exception as e:
        st.warning("Impossible de charger les données pour le moment. Vérifiez votre connexion ou le fichier `data.py`.")

st.divider()

# =========================================================
# 2. Dictionnaire des Variables
# =========================================================
st.header("2. Dictionnaire des variables")
st.markdown("Pour comprendre comment notre Intelligence Artificielle \"écoute\" la musique, voici la signification technique de chaque colonne présente dans la base de données :")

# Création du dictionnaire de données comme demandé dans l'énoncé
data_dict = {
    "Variable": [
        "track_id, track_name, track_artist",
        "track_popularity",
        "track_album_id, track_album_name, track_album_release_date",
        "playlist_name, playlist_id",
        "playlist_genre",
        "playlist_subgenre",
        "danceability, energy, speechiness, acousticness, instrumentalness, liveness, valence",
        "loudness",
        "tempo",
        "key, mode",
        "duration_ms"
    ],
    "Description": [
        "Identifiant unique, titre et artiste du morceau.",
        "Score de popularité Spotify (de 0 à 100).",
        "Identifiant, nom et date de sortie de l'album d'origine.",
        "Nom et identifiant de la playlist contenant le morceau.",
        "Genre musical principal (edm, latin, pop, r&b, rap, rock). 🎯 Cible de la partie Classification.",
        "Sous-genre musical plus précis.",
        "Indicateurs audio techniques (compris entre 0 et 1). 🎯 'energy' est la cible de la partie Régression.",
        "Volume moyen de la piste (en décibels - dB).",
        "Vitesse et rythme du morceau (en BPM - Battements par minute).",
        "Tonalité (de 0 à 11) et Mode (1 = majeur, 0 = mineur).",
        "Durée brute du morceau en millisecondes."
    ]
}

# Affichage avec st.table pour un rendu statique très propre
df_dict = pd.DataFrame(data_dict)
st.table(df_dict)