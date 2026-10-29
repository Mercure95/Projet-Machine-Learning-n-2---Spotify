<<<<<<< HEAD
=======
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
import seaborn as sns
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Import des configurations communes
from config import RANDOM_STATE, TEST_SIZE, TARGET_REG, FEATURES_REG
from data import load_data, clean_data

st.set_page_config(page_title="Régression - Énergie", page_icon="📈", layout="wide")
st.title("📈 Prédire l'énergie d'un morceau")

# =========================================================
# 1. Démarche et Informations sur le modèle
# =========================================================
st.header("1. Notre démarche")
st.markdown(f"""
Nous cherchons à prédire la variable cible **`{TARGET_REG}`** (continue entre 0 et 1) à partir des caractéristiques audio des morceaux.
* **Algorithme utilisé** : Régression Linéaire (Simple puis Multiple).
* **Variables explicatives (X)** : `{', '.join(FEATURES_REG)}`.
* **Préparation** : Les données ont été nettoyées, et séparées en un jeu d'entraînement (Train, {int((1-TEST_SIZE)*100)}%) et un jeu de test (Test, {int(TEST_SIZE*100)}%) avec une graine aléatoire fixée à {RANDOM_STATE} pour assurer la reproductibilité.
""")

# =========================================================
# Chargement des données et des modèles en cache
# =========================================================
@st.cache_data
def get_data_and_metrics():
    # Chargement et préparation des données
    df = load_data()
    df_clean = clean_data(df)
    
    X = df_clean[FEATURES_REG]
    y = df_clean[TARGET_REG]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Régression Simple (Loudness uniquement)
    X_train_s = X_train[['loudness']]
    X_test_s = X_test[['loudness']]
    model_simple = LinearRegression()
    model_simple.fit(X_train_s, y_train)
    
    return X_train, X_test, y_train, y_test, model_simple

# On exécute la fonction
X_train, X_test, y_train, y_test, model_simple = get_data_and_metrics()

# Chargement du modèle multiple final sauvegardé depuis le notebook
@st.cache_resource
def load_final_model():
    return joblib.load('models/regression.joblib')

try:
    model_multi = load_final_model()
except FileNotFoundError:
    st.error("Le modèle n'a pas été trouvé. Exécutez le notebook 02_regression.ipynb pour générer 'models/regression.joblib'.")
    st.stop()

# =========================================================
# 2. Performances et Comparaison
# =========================================================
st.header("2. Évaluation des performances")

# Fonction locale pour calculer les métriques
def calc_metrics(model, X_tr, y_tr, X_te, y_te):
    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)
    return {
        'R² Train': round(r2_score(y_tr, pred_tr), 3),
        'R² Test': round(r2_score(y_te, pred_te), 3),
        'MAE Train': round(mean_absolute_error(y_tr, pred_tr), 3),
        'MAE Test': round(mean_absolute_error(y_te, pred_te), 3),
        'RMSE Test': round(np.sqrt(mean_squared_error(y_te, pred_te)), 3)
    }

# Calcul des métriques pour les deux modèles
metrics_simple = calc_metrics(model_simple, X_train[['loudness']], y_train, X_test[['loudness']], y_test)
metrics_multi = calc_metrics(model_multi, X_train, y_train, X_test, y_test)

# Affichage du tableau comparatif
df_metrics = pd.DataFrame([metrics_simple, metrics_multi], index=["Régression Simple (Loudness)", "Régression Multiple"])
st.dataframe(df_metrics, use_container_width=True)

st.markdown("""
**Interprétation :**
* Les scores sur le jeu Train et le jeu Test sont très proches : notre modèle ne fait pas d'overfitting.
* La régression multiple est nettement supérieure à la régression simple, car le coefficient de détermination (R²) augmente et l'erreur moyenne (MAE) diminue.
""")

# =========================================================
# 3. Interprétation Visuelle (Graphiques)
# =========================================================
st.header("3. Analyse visuelle du Modèle Multiple")

tab1, tab2 = st.tabs(["Valeurs Réelles vs Prédites", "Coefficients du modèle"])

with tab1:
    y_pred_multi = model_multi.predict(X_test)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=y_test, y=y_pred_multi, alpha=0.3, color='royalblue', ax=ax1)
    ax1.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2)
    ax1.set_title('Régression Multiple : Énergie Réelle vs Prédite')
    ax1.set_xlabel('Énergie Réelle (Test)')
    ax1.set_ylabel('Énergie Prédite (Test)')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    st.pyplot(fig1)

with tab2:
    coefficients = pd.DataFrame(
        list(zip(FEATURES_REG, model_multi.coef_)), 
        columns=['Variable', 'Coefficient']
    )
    coefficients['Impact Absolu'] = coefficients['Coefficient'].abs()
    coefficients = coefficients.sort_values(by='Impact Absolu', ascending=False)
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.barplot(data=coefficients, x='Coefficient', y='Variable', palette='coolwarm', ax=ax2)
    ax2.set_title("Poids des variables dans la prédiction de l'énergie")
    st.pyplot(fig2)

st.divider()

# =========================================================
# 4. Formulaire interactif
# =========================================================
st.header("4. Testez le modèle vous-même 🎛️")
st.markdown("Ajustez les curseurs pour voir comment le modèle prédit l'énergie d'un morceau en temps réel.")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
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

    submit_button = st.form_submit_button(label="🔮 Calculer l'énergie")

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
    
    st.success(f"### Énergie prédite : {prediction_bounded:.2f} / 1.00")
    st.progress(float(prediction_bounded))
>>>>>>> fb37c975811ab2c2c64ebdff8d19f4396f74ba51
