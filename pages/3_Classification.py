import streamlit as st
import pandas as pd
import numpy as np
import joblib
from data import load_data, clean_data

st.set_page_config(page_title="Classification - Genre", page_icon="🎵", layout="wide")

st.title("🎵 Prédire le genre musical d'un morceau")

# --- 1. PRÉSENTATION DU MODÈLE ---
st.header("1. Performances du modèle")
st.markdown("""
*En attente des résultats de l'équipe ML.*
* **Modèle choisi :** Régression Logistique / KNN / Random Forest
* **Variable cible :** `playlist_genre` (6 classes : edm, latin, pop, r&b, rap, rock)
* **Variables explicatives :** Caractéristiques audio, durée, année
""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy (Test)", "À venir...")
col2.metric("Précision", "À venir...")
col3.metric("Rappel", "À venir...")
col4.metric("Score F1", "À venir...")

st.info("Ici viendra la matrice de confusion (Confusion Matrix) générée par l'équipe.")
st.divider()

# Chargement des données pour le sélecteur
df = clean_data(load_data())
genres_cibles = ['edm', 'latin', 'pop', 'r&b', 'rap', 'rock']

# --- 2. FORMULAIRE DE PRÉDICTION ---
st.header("2. Testez le modèle 🎛️")

# Système d'onglets pour séparer le test manuel du test sur morceau existant
tab1, tab2 = st.tabs(["Saisie manuelle", "Choisir un morceau existant"])

with tab1:
    with st.form("classification_form"):
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            danceability = st.slider("Danceability", 0.0, 1.0, 0.5, key="c_dance")
            energy = st.slider("Energy", 0.0, 1.0, 0.7, key="c_nrj") # Ajouté pour la classification
            speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1, key="c_speech")
            acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5, key="c_acous")
            instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0, key="c_instru")
            
        with col_b:
            liveness = st.slider("Liveness", 0.0, 1.0, 0.2, key="c_live")
            valence = st.slider("Valence (Positivité)", 0.0, 1.0, 0.5, key="c_val")
            loudness = st.slider("Loudness (dB)", -60.0, 0.0, -10.0, key="c_loud")
            tempo = st.slider("Tempo (BPM)", 0.0, 250.0, 120.0, key="c_tempo")
            
        with col_c:
            key_val = st.selectbox("Tonalité (Key)", options=list(range(12)), key="c_key")
            mode = st.selectbox("Mode (1=Majeur, 0=Mineur)", options=[0, 1], key="c_mode")
            duration_min = st.number_input("Durée (minutes)", min_value=0.5, max_value=20.0, value=3.5, key="c_dur")
            release_year = st.number_input("Année de sortie", min_value=1950, max_value=2026, value=2020, key="c_year")

        submit_manual = st.form_submit_button(label="🔮 Prédire le genre")

with tab2:
    titres_dispos = df['track_name'].dropna().unique()
    titre_choisi = st.selectbox("Rechercher un morceau du catalogue :", titres_dispos)
    submit_auto = st.button("🔮 Analyser ce morceau")

# --- 3. LOGIQUE DE PRÉDICTION (MOCKING) ---
if submit_manual or submit_auto:
    if submit_auto:
        # Récupération des données du morceau sélectionné
        morceau_data = df[df['track_name'] == titre_choisi].iloc[0]
        st.write(f"**Genre réel du morceau dans le dataset :** {morceau_data['playlist_genre'].upper()}")
        
    # ⚠️ MOCKING : Fausse prédiction de probabilités (somme = 1)
    probas = np.random.dirichlet(np.ones(6), size=1)[0]
    df_probas = pd.DataFrame({"Probabilités": probas}, index=genres_cibles)
    genre_predit = genres_cibles[np.argmax(probas)]
    
    # -- CODE FINAL À DÉCOMMENTER QUAND LE MODÈLE SERA PRÊT --
    # model_clf = joblib.load("models/classification.joblib")
    # probas = model_clf.predict_proba(user_data)[0]
    # classes = model_clf.classes_
    # df_probas = pd.DataFrame({"Probabilités": probas}, index=classes)
    # genre_predit = classes[np.argmax(probas)]
    
    st.success(f"🎵 Genre musical prédit : **{genre_predit.upper()}**")
    
    # Affichage du graphique en barres des probabilités
    st.subheader("Détail des probabilités par genre")
    st.bar_chart(df_probas)