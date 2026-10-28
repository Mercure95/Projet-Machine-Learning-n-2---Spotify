import streamlit as st
from data import load_data, clean_data

st.set_page_config(page_title="Spotify ML Project", page_icon="🔥", layout="wide")

st.title("ᯤ Spotify Décrypter la musique avec Spotify")
st.markdown("""
**Contexte du projet :**
Une start-up de la musique souhaite analyser automatiquement les morceaux à partir de leurs caractéristiques audio via l'API Spotify. 
Notre objectif est double :
1. **Régression :** Prédire l'énergie d'un morceau.
2. **Classification :** Prédire le genre musical d'un morceau.

*Projet réalisé par : GOASDUFF Pierre, MARILLEAU Bastien, MARCHAND Erwann, SOICHET Quentin*
""")

st.divider()

with st.spinner("Chargement des données en cours..."):
    df_brut = load_data()
    df = clean_data(df_brut)

st.subheader("Aperçu du jeu de données")
col1, col2 = st.columns(2)
col1.metric("Nombre de morceaux (lignes)", df.shape[0])
col2.metric("Nombre de variables (colonnes)", df.shape[1])

st.dataframe(df.head(10))

st.subheader("Dictionnaire des variables")
st.markdown("""
* **playlist_genre** : Cible classification (edm, latin, pop, r&b, rap, rock).
* **energy** : Cible régression (0.0 à 1.0).
* **danceability, speechiness, acousticness, instrumentalness, liveness, valence** : Indicateurs audio entre 0 et 1.
* **loudness** : Volume moyen (dB).
* **tempo** : Tempo (BPM).
* **duration_min** : Durée de la piste en minutes.
""")