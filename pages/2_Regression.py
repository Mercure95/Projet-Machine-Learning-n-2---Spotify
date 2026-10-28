import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Régression - Énergie", page_icon="📈", layout="wide")

st.title("📈 Prédire l'énergie d'un morceau")

# --- 1. PRÉSENTATION DU MODÈLE ---
st.header("1. Performances du modèle")
st.markdown("""
*En attente des résultats de l'équipe ML.*
* **Modèle choisi :** Régression Linéaire Multiple / Random Forest
* **Variables explicatives :** Caractéristiques audio, durée, année
* **Préparation :** Standardisation, encodage...
""")

col1, col2, col3 = st.columns(3)
col1.metric("R² (Test)", "À venir...")
col2.metric("MAE (Test)", "À venir...")
col3.metric("RMSE (Test)", "À venir...")

st.info("Ici viendra le graphique des valeurs réelles vs prédites et l'affichage des coefficients.")

st.divider()

# --- 2. FORMULAIRE DE PRÉDICTION ---
st.header("2. Testez le modèle : à vous de mixer ! 🎛️")
st.write("Ajustez les paramètres audio ci-dessous pour prédire l'énergie (`energy`) du morceau.")

# Création du formulaire pour éviter que la page se recharge à chaque mouvement de curseur
with st.form("regression_form"):
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
        
    with col_b:
        liveness = st.slider("Liveness", 0.0, 1.0, 0.2)
        valence = st.slider("Valence (Positivité)", 0.0, 1.0, 0.5)
        loudness = st.slider("Loudness (dB)", -60.0, 0.0, -10.0)
        tempo = st.slider("Tempo (BPM)", 0.0, 250.0, 120.0)
        
    with col_c:
        key = st.selectbox("Tonalité (Key)", options=list(range(12)))
        mode = st.selectbox("Mode (1=Majeur, 0=Mineur)", options=[0, 1])
        duration_min = st.number_input("Durée (minutes)", min_value=0.5, max_value=20.0, value=3.5)
        release_year = st.number_input("Année de sortie", min_value=1950, max_value=2026, value=2020)

    # Bouton de validation du formulaire
    submit_button = st.form_submit_button(label="🔮 Prédire l'énergie")

# --- 3. RÉSULTAT DE LA PRÉDICTION ---
if submit_button:
    # On rassemble les inputs exactement dans le même ordre que FEATURES_REG (cf. variables.txt)
    user_data = pd.DataFrame([{
        "danceability": danceability,
        "speechiness": speechiness,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "valence": valence,
        "loudness": loudness,
        "tempo": tempo,
        "key": key,
        "mode": mode,
        "duration_min": duration_min,
        "release_year": release_year
    }])
    
    # ⚠️ MOCKING : Fausse prédiction en attendant le vrai modèle
    prediction_simulee = np.random.uniform(0.4, 0.9)
    
    # -- CODE FINAL À DÉCOMMENTER QUAND LE MODÈLE SERA PRÊT --
    # try:
    #     model_reg = joblib.load("models/regression.joblib")
    #     prediction = model_reg.predict(user_data)[0]
    # except FileNotFoundError:
    #     st.error("Le fichier du modèle (models/regression.joblib) est introuvable.")
    #     prediction = prediction_simulee
    
    st.success(f"⚡ Énergie prédite : **{prediction_simulee:.2f}** / 1.00")
    st.progress(prediction_simulee)