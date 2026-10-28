# =========================================================
# Configuration des chemins d'importation
# =========================================================
import sys
import os

# Permet à Python de trouver les fichiers config.py et data.py à la racine du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np

# Import des configurations communes depuis la racine
from config import FEATURES_REG

st.set_page_config(page_title="Régression - Énergie", page_icon="📈")
st.title("📈 Prédire l'énergie d'un morceau")

st.markdown("""
Cette page présente notre modèle de **Régression Linéaire Multiple**. 
L'objectif est de prédire la variable continue `energy` (comprise entre 0 et 1) en utilisant les autres caractéristiques audio.
""")

# =========================================================
# Chargement du modèle
# =========================================================
@st.cache_resource
def load_model():
    # Streamlit est exécuté depuis la racine, on pointe donc vers le dossier models/
    return joblib.load('models/regression.joblib')

try:
    model_reg_multi = load_model()
except FileNotFoundError:
    st.error("Le modèle n'a pas été trouvé. Veuillez exécuter le notebook 02_regression.ipynb au préalable pour générer 'models/regression.joblib'.")
    st.stop()

# =========================================================
# Formulaire de prédiction interactif
# =========================================================
st.header("🎛️ Testez le modèle")
st.markdown("Ajustez les paramètres ci-dessous pour estimer l'énergie d'un morceau fictif.")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    # Répartition des inputs
    with col1:
        loudness = st.slider("Loudness (dB)", min_value=-60.0, max_value=5.0, value=-5.0, step=0.5)
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.2, step=0.05)
        danceability = st.slider("Danceability", 0.0, 1.0, 0.6, step=0.05)
        tempo = st.number_input("Tempo (BPM)", min_value=0.0, max_value=250.0, value=120.0)
        
    with col2:
        valence = st.slider("Valence", 0.0, 1.0, 0.5, step=0.05)
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0, step=0.05)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.15, step=0.05)
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05, step=0.05)
        
    with col3:
        key = st.selectbox("Key (Tonalité)", options=list(range(12)))
        mode = st.selectbox("Mode", options=[0, 1], format_func=lambda x: "Majeur (1)" if x == 1 else "Mineur (0)")
        duration_min = st.number_input("Durée (minutes)", min_value=0.5, max_value=15.0, value=3.5, step=0.1)
        release_year = st.number_input("Année de sortie", min_value=1950, max_value=2024, value=2020, step=1)

    submit_button = st.form_submit_button(label="🔮 Prédire l'énergie")

# =========================================================
# Traitement de la prédiction
# =========================================================
if submit_button:
    # Création d'un DataFrame avec 1 ligne contenant les valeurs saisies par l'utilisateur
    input_data = {
        'danceability': [danceability],
        'speechiness': [speechiness],
        'acousticness': [acousticness],
        'instrumentalness': [instrumentalness],
        'liveness': [liveness],
        'valence': [valence],
        'loudness': [loudness],
        'tempo': [tempo],
        'key': [key],
        'mode': [mode],
        'duration_min': [duration_min],
        'release_year': [release_year]
    }
    
    # On réorganise les colonnes pour respecter STRICTEMENT l'ordre attendu par le modèle
    X_user = pd.DataFrame(input_data)[FEATURES_REG]
    
    # Prédiction
    prediction = model_reg_multi.predict(X_user)[0]
    
    # Borner la prédiction entre 0 et 1 (au cas où la régression linéaire donne un résultat aberrant)
    prediction_bounded = np.clip(prediction, 0, 1)
    
    st.success(f"### Énergie prédite : {prediction_bounded:.2f}")
    
    # Affichage visuel sous forme de jauge (progress bar)
    st.progress(float(prediction_bounded))