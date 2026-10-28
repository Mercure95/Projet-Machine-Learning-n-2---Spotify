import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from data import load_data, clean_data

st.set_page_config(page_title="Exploration des données", page_icon="📊", layout="wide")

st.title("📊 Analyse exploratoire des données (EDA)")

# Chargement des données via les fonctions communes
df_brut = load_data()
df = clean_data(df_brut)

# Filtre interactif demandé par l'énoncé
st.sidebar.header("Filtres interactifs")
genres_dispos = df['playlist_genre'].unique()
genre_selectionne = st.sidebar.multiselect(
    "Sélectionnez un ou plusieurs genres :", 
    options=genres_dispos, 
    default=genres_dispos
)

# Application du filtre
df_filtre = df[df['playlist_genre'].isin(genre_selectionne)]

st.subheader("Indicateurs clés du sous-échantillon")
col1, col2, col3 = st.columns(3)
col1.metric("Morceaux", df_filtre.shape[0])
col2.metric("Durée moyenne (min)", round(df_filtre['duration_min'].mean(), 2))
col3.metric("Énergie moyenne", round(df_filtre['energy'].mean(), 2))

st.divider()

st.subheader("1. Distribution des caractéristiques audio (En attente)")
st.info("Ici, nous intégrerons l'histogramme des distributions (issu de la partie EDA).")
# st.pyplot(fig1) # À décommenter quand l'équipe te fournira la figure

st.subheader("2. Corrélation entre les variables (En attente)")
st.info("Ici, nous intégrerons la heatmap des corrélations.")
# st.pyplot(fig2)

st.subheader("Synthèse et constats clés")
st.markdown("""
*En attente des 5 constats clés rédigés par l'équipe en charge de l'EDA.*
""")