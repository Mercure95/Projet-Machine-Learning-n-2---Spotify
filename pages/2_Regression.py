# =========================================================
# Configuration des chemins d'importation
# =========================================================
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from config import RANDOM_STATE, TEST_SIZE, TARGET_REG, FEATURES_REG
from data import load_data, clean_data

st.set_page_config(page_title="Régression - Énergie", page_icon="📈", layout="wide")
st.title("📈 Prédire l'énergie d'un morceau : Démarche et Modélisation")

st.markdown("""
Cette page détaille la construction de notre modèle de **Régression Linéaire**. 
Notre objectif est de prédire la variable cible **`energy`** (l'intensité du morceau, de 0 à 1) à partir de ses autres caractéristiques audio, sans jamais tricher en regardant la réponse.
""")

# =========================================================
# Chargement des données et préparation en cache
# =========================================================
@st.cache_data
def prepare_all_models():
    df = load_data()
    df_clean = clean_data(df)
    
    # --- Filtre A : Loudness ---
    df_loud = df_clean[(df_clean['duration_min'] >= 1.0) & (df_clean['duration_min'] <= 10.0) & (df_clean['loudness'] >= -20.0)]
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(df_loud[['loudness']], df_loud[TARGET_REG], test_size=TEST_SIZE, random_state=RANDOM_STATE)
    model_loud = LinearRegression().fit(X_train_l, y_train_l)
    
    # --- Filtre B : Acousticness ---
    df_aco = df_clean[(df_clean['duration_min'] >= 1.0) & (df_clean['duration_min'] <= 10.0) & (df_clean['speechiness'] < 0.5)]
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(df_aco[['acousticness']], df_aco[TARGET_REG], test_size=TEST_SIZE, random_state=RANDOM_STATE)
    model_aco = LinearRegression().fit(X_train_a, y_train_a)
    
    # --- Filtre Global : Multiple ---
    df_multi = df_clean[(df_clean['duration_min'] >= 1.0) & (df_clean['duration_min'] <= 10.0) & (df_clean['loudness'] >= -20.0) & (df_clean['tempo'] > 0) & (df_clean['speechiness'] < 0.5)]
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(df_multi[FEATURES_REG], df_multi[TARGET_REG], test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    return df_clean, df_loud, df_aco, df_multi, model_loud, X_test_l, y_test_l, model_aco, X_test_a, y_test_a, X_test_m, y_test_m

(df_initial, df_loud, df_aco, df_multi, 
 model_loud, X_test_l, y_test_l, 
 model_aco, X_test_a, y_test_a, 
 X_test_m, y_test_m) = prepare_all_models()

@st.cache_resource
def load_final_model():
    return joblib.load('models/regression.joblib')

try:
    model_multi = load_final_model()
except FileNotFoundError:
    st.error("Modèle introuvable. Exécutez le notebook 02_regression.ipynb pour le générer.")
    st.stop()

# =========================================================
# Section 1 : Préparation et Filtrages
# =========================================================
st.header("1. La préparation des données (Filtrages ciblés)")
st.markdown("""
La régression linéaire trace une ligne au milieu d'un nuage de points. Elle est donc très sensible aux valeurs extrêmes (les *outliers*). Nous avons testé deux variables individuellement avant de les combiner, en appliquant un filtrage spécifique à chaque fois.
""")

col1, col2 = st.columns(2)
with col1:
    st.info(f"**Filtre A : Centré sur le Volume (Loudness)**\n\nNous avons retiré les morceaux avec un volume inférieur à -20 dB (quasi-silences) et les durées aberrantes. \n\n*Volume conservé : {len(df_loud)} morceaux.*")
with col2:
    st.info(f"**Filtre B : Centré sur l'Acoustique (Acousticness)**\n\nNous avons retiré les morceaux très parlés (`speechiness > 0.5`), car les podcasts perturbent la corrélation musicale. \n\n*Volume conservé : {len(df_aco)} morceaux.*")

st.markdown(f"**Filtre Global pour le modèle final** : Pour le modèle multiple, nous avons combiné toutes ces règles. Notre jeu de données final, parfaitement propre, contient **{len(df_multi)}** morceaux sur les {len(df_initial)} initiaux.")

st.divider()

# =========================================================
# Section 2 : Comparaison des Modèles
# =========================================================
st.header("2. Analyse et Comparaison des Modèles")

def calc_test_metrics(model, X_te, y_te):
    pred_te = model.predict(X_te)
    return {
        'R² (Explication de la variance)': f"{r2_score(y_te, pred_te):.3f}",
        'MAE (Erreur Moyenne)': f"{mean_absolute_error(y_te, pred_te):.3f}",
        'RMSE (Erreur Quadratique)': f"{np.sqrt(mean_squared_error(y_te, pred_te)):.3f}"
    }

df_metrics = pd.DataFrame([
    calc_test_metrics(model_loud, X_test_l, y_test_l),
    calc_test_metrics(model_aco, X_test_a, y_test_a),
    calc_test_metrics(model_multi, X_test_m, y_test_m)
], index=["1. Simple (Loudness)", "2. Simple (Acousticness)", "3. Multiple (Toutes variables)"])

st.table(df_metrics)

st.markdown("""
**Que nous apprennent ces résultats ?**
* Le **Modèle 1** montre que le volume seul explique déjà une grande partie de l'énergie (R² élevé).
* Le **Modèle 2** confirme que l'acoustique est un bon indicateur, mais moins précis que le volume.
* Le **Modèle 3 (Multiple)** combine les forces de toutes les caractéristiques. Il atteint le meilleur score R² et minimise l'erreur absolue (MAE).
""")

tab1, tab2 = st.tabs(["📊 Nuage de points (Modèle Multiple)", "⚖️ Poids des variables (Coefficients)"])

with tab1:
    y_pred_multi = model_multi.predict(X_test_m)
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.scatterplot(x=y_test_m, y=y_pred_multi, alpha=0.3, color='royalblue', ax=ax1)
    ax1.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2)
    ax1.set_title("Modèle Multiple : Énergie Réelle vs Énergie Prédite")
    ax1.set_xlabel("Vraie Énergie (Jeu de test)")
    ax1.set_ylabel("Énergie devinée par le Modèle")
    st.pyplot(fig1)

with tab2:
    coefficients = pd.DataFrame(list(zip(FEATURES_REG, model_multi.coef_)), columns=['Variable', 'Coefficient'])
    coefficients['Impact Absolu'] = coefficients['Coefficient'].abs()
    coefficients = coefficients.sort_values(by='Impact Absolu', ascending=False)
    
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=coefficients, x='Coefficient', y='Variable', palette='vlag', ax=ax2)
    ax2.set_title("Impact de chaque variable sur le calcul de l'énergie")
    st.pyplot(fig2)

st.divider()

# =========================================================
# Section 3 : Simulateur interactif
# =========================================================
st.header("3. Testez le modèle Multiple : Le Simulateur 🎛️")
st.markdown("Modifiez les caractéristiques ci-dessous et observez comment l'équation mathématique réajuste l'énergie en direct.")

with st.form("prediction_form"):
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        loudness = st.slider("Loudness (Volume en dB)", -30.0, 5.0, -5.0, 0.5)
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.2, 0.05)
        danceability = st.slider("Danceability", 0.0, 1.0, 0.6, 0.05)
        tempo = st.number_input("Tempo (BPM)", 0.0, 250.0, 120.0)
        
    with col_s2:
        valence = st.slider("Valence (Positivité)", 0.0, 1.0, 0.5, 0.05)
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0, 0.05)
        liveness = st.slider("Liveness (Concert)", 0.0, 1.0, 0.15, 0.05)
        speechiness = st.slider("Speechiness (Paroles)", 0.0, 1.0, 0.05, 0.05)
        
    with col_s3:
        key = st.selectbox("Key (Tonalité 0-11)", options=list(range(12)))
        mode = st.selectbox("Mode", options=[0, 1], format_func=lambda x: "Majeur (1)" if x == 1 else "Mineur (0)")
        duration_min = st.number_input("Durée (minutes)", 0.5, 15.0, 3.5, 0.1)
        release_year = st.number_input("Année de sortie", 1950, 2024, 2020, 1)

    submit_button = st.form_submit_button(label="🔮 Demander au modèle de prédire l'énergie")

if submit_button:
    input_data = {
        'danceability': [danceability], 'speechiness': [speechiness],
        'acousticness': [acousticness], 'instrumentalness': [instrumentalness],
        'liveness': [liveness], 'valence': [valence],
        'loudness': [loudness], 'tempo': [tempo],
        'key': [key], 'mode': [mode],
        'duration_min': [duration_min], 'release_year': [release_year]
    }
    X_user = pd.DataFrame(input_data)[FEATURES_REG]
    prediction = model_multi.predict(X_user)[0]
    
    prediction_bounded = np.clip(prediction, 0, 1)
    
    st.success(f"### ⚡ Énergie calculée par l'algorithme : {prediction_bounded:.2f} / 1.00")
    st.progress(float(prediction_bounded))