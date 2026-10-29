# =========================================================
# Configuration des chemins d'importation
# =========================================================
import sys
import os

# Permet à Python de trouver les fichiers config.py et data.py à la racine
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
st.title("📈 Prédire l'énergie d'un morceau : Démarche et Modélisation")

# =========================================================
# 0. L'accroche et l'objectif (Nouveau)
# =========================================================
st.info("🎯 **Notre défi :** Peut-on deviner mathématiquement l'énergie d'un morceau de musique sans jamais l'écouter, uniquement en analysant ses caractéristiques techniques ?")

st.markdown("""
Cette page détaille la construction de notre algorithme de **Régression Linéaire** pour répondre à cette question. 
Notre objectif précis est de prédire la variable cible **`energy`** (l'intensité globale d'une piste, notée de 0 à 1) en croisant des données mathématiques brutes comme le volume, le tempo ou l'acoustique.

**Pour concevoir un modèle robuste, notre démarche suit 3 étapes :**
1. **Un nettoyage minutieux** pour écarter les données absurdes.
2. **Des modèles simples** (à une variable) pour valider nos hypothèses de départ.
3. **Un modèle multiple** (toutes les variables) pour maximiser la précision.
""")
st.divider()

# =========================================================
# Chargement des données et préparation en cache
# =========================================================
@st.cache_data
def prepare_all_models():
    df = load_data()
    df_clean = clean_data(df)
    
    # Filtre A : Loudness
    df_loud = df_clean[(df_clean['duration_min'] >= 1.0) & (df_clean['duration_min'] <= 10.0) & (df_clean['loudness'] >= -20.0)]
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(df_loud[['loudness']], df_loud[TARGET_REG], test_size=TEST_SIZE, random_state=RANDOM_STATE)
    model_loud = LinearRegression().fit(X_train_l, y_train_l)
    
    # Filtre B : Acousticness
    df_aco = df_clean[(df_clean['duration_min'] >= 1.0) & (df_clean['duration_min'] <= 10.0) & (df_clean['speechiness'] < 0.5)]
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(df_aco[['acousticness']], df_aco[TARGET_REG], test_size=TEST_SIZE, random_state=RANDOM_STATE)
    model_aco = LinearRegression().fit(X_train_a, y_train_a)
    
    # Filtre Global : Multiple
    df_multi = df_clean[(df_clean['duration_min'] >= 1.0) & (df_clean['duration_min'] <= 10.0) & (df_clean['loudness'] >= -20.0) & (df_clean['tempo'] > 0) & (df_clean['speechiness'] < 0.5)]
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(df_multi[FEATURES_REG], df_multi[TARGET_REG], test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    return df_clean, df_loud, df_aco, df_multi, model_loud, X_train_l, X_test_l, y_train_l, y_test_l, model_aco, X_train_a, X_test_a, y_train_a, y_test_a, X_train_m, X_test_m, y_train_m, y_test_m

(df_initial, df_loud, df_aco, df_multi, 
 model_loud, X_train_l, X_test_l, y_train_l, y_test_l, 
 model_aco, X_train_a, X_test_a, y_train_a, y_test_a, 
 X_train_m, X_test_m, y_train_m, y_test_m) = prepare_all_models()

@st.cache_resource
def load_final_model():
    return joblib.load('models/regression.joblib')

try:
    model_multi = load_final_model()
except FileNotFoundError:
    st.error("Modèle introuvable. Exécutez le notebook 02_regression.ipynb pour le générer.")
    st.stop()

# =========================================================
# 1. Préparation, Filtrage et Train/Test Split
# =========================================================
st.header("1. Préparation des données et concept de Train/Test Split")

st.markdown("""
La régression linéaire trace une droite au cœur d'un nuage de points. Elle est mathématiquement très vulnérable aux valeurs aberrantes (outliers). Avant d'entraîner nos modèles, nous appliquons un **pré-filtrage** spécifique selon la variable étudiée pour ne conserver que la "vraie" musique.
""")

st.code("""
# Exemple de filtrage global pour le modèle multiple
df_multi = df_clean[
    (df_clean['duration_min'] >= 1.0) & (df_clean['duration_min'] <= 10.0) & # Exclut les jingles ou mix de 2h
    (df_clean['loudness'] >= -20.0) &                                        # Exclut les quasi-silences
    (df_clean['tempo'] > 0) &                                                # Exclut les erreurs d'API
    (df_clean['speechiness'] < 0.5)                                          # Exclut les podcasts (trop parlés)
]
""", language="python")

st.markdown("""
### La séparation Train / Test
Pour vérifier qu'un étudiant a compris son cours, on ne lui donne pas les exercices qu'il a déjà appris par cœur : on lui donne un examen inédit. En Machine Learning, c'est la fonction `train_test_split`.
* **Train (80%)** : Le modèle observe les caractéristiques audio et l'énergie réelle, puis calcule ses coefficients.
* **Test (20%)** : Le modèle découvre de nouveaux morceaux. On lui cache l'énergie, il doit la prédire, et on mesure son erreur.
""")

st.code("""
# Découpage avec une graine fixe (random_state=42) pour la reproductibilité
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)
""", language="python")

st.divider()

# Fonction utilitaire pour générer les graphiques de dispersion
def plot_scatter(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.3, color='royalblue', ax=ax)
    ax.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Vraie Énergie")
    ax.set_ylabel("Énergie Prédite")
    return fig

# =========================================================
# 2. Modèle Simple A (Loudness)
# =========================================================
st.header("2. Modèle Simple A : Prédiction par le Volume (`loudness`)")
st.markdown("""
Le `loudness` possède la plus forte corrélation positive avec l'énergie. Nous avons filtré les silences extrêmes (`loudness < -20 dB`).
""")

st.code("""
# Entraînement sur une seule dimension (X_train_l ne contient que 'loudness')
model_loud = LinearRegression()
model_loud.fit(X_train_l, y_train_l)
""", language="python")

col_a1, col_a2 = st.columns([1, 2])
y_pred_l = model_loud.predict(X_test_l)
with col_a1:
    st.metric("R² Test", f"{r2_score(y_test_l, y_pred_l):.3f}")
    st.metric("MAE Test", f"{mean_absolute_error(y_test_l, y_pred_l):.3f}")
    st.markdown("**Interprétation :** Le volume seul parvient déjà à expliquer une part importante de l'énergie. Le nuage de points suit la ligne rouge, mais reste très dispersé.")
with col_a2:
    st.pyplot(plot_scatter(y_test_l, y_pred_l, "Modèle Simple A (Loudness)"))

st.divider()

# =========================================================
# 3. Modèle Simple B (Acousticness)
# =========================================================
st.header("3. Modèle Simple B : Prédiction par l'Acoustique (`acousticness`)")
st.markdown("""
L'`acousticness` possède la plus forte corrélation négative (un morceau très acoustique manque d'énergie). Nous avons exclu les podcasts (`speechiness > 0.5`) pour ne pas fausser l'analyse.
""")

st.code("""
# Entraînement sur la dimension acoustique
model_aco = LinearRegression()
model_aco.fit(X_train_a, y_train_a)
""", language="python")

col_b1, col_b2 = st.columns([1, 2])
y_pred_a = model_aco.predict(X_test_a)
with col_b1:
    st.metric("R² Test", f"{r2_score(y_test_a, y_pred_a):.3f}")
    st.metric("MAE Test", f"{mean_absolute_error(y_test_a, y_pred_a):.3f}")
    st.markdown("**Interprétation :** L'acoustique est un bon indicateur de faible énergie, mais elle est très insuffisante pour estimer avec précision l'énergie d'un morceau complexe.")
with col_b2:
    st.pyplot(plot_scatter(y_test_a, y_pred_a, "Modèle Simple B (Acousticness)"))

st.divider()

# =========================================================
# 4. Modèle Multiple
# =========================================================
st.header("4. Modèle Multiple : La puissance de toutes les variables")
st.markdown("""
Nous utilisons ici toutes nos caractéristiques audio (`FEATURES_REG`) sur un catalogue trié grâce au filtre global. L'algorithme calcule un hyperplan multidimensionnel qui pondère chaque variable.
""")

st.code("""
# Entraînement sur toutes les variables (X_train_m)
model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)

# Récupération des coefficients pour l'analyse
poids_variables = model_multi.coef_
""", language="python")

col_m1, col_m2 = st.columns([1, 2])
y_pred_m = model_multi.predict(X_test_m)
with col_m1:
    st.metric("R² Test", f"{r2_score(y_test_m, y_pred_m):.3f}")
    st.metric("MAE Test", f"{mean_absolute_error(y_test_m, y_pred_m):.3f}")
    st.markdown("**Interprétation :** Les erreurs s'annulent. En croisant le volume avec le tempo ou l'instrumentalité, le modèle gagne drastiquement en précision. Le nuage de points est nettement plus resserré sur la ligne de prédiction parfaite.")
with col_m2:
    st.pyplot(plot_scatter(y_test_m, y_pred_m, "Modèle Multiple"))

# Affichage des coefficients
st.subheader("Analyse des coefficients du modèle multiple")
coefficients = pd.DataFrame(list(zip(FEATURES_REG, model_multi.coef_)), columns=['Variable', 'Coefficient'])
coefficients['Impact Absolu'] = coefficients['Coefficient'].abs()
coefficients = coefficients.sort_values(by='Impact Absolu', ascending=False)

fig_coef, ax_coef = plt.subplots(figsize=(10, 3))
sns.barplot(data=coefficients, x='Coefficient', y='Variable', palette='vlag', ax=ax_coef)
ax_coef.set_title("Poids mathématique de chaque variable dans l'équation finale")
st.pyplot(fig_coef)

st.divider()

# =========================================================
# 5. Comparaison Globale
# =========================================================
st.header("5. Comparaison et Bilan")
st.markdown("Récapitulatif des erreurs sur le jeu de données de Test (données inconnues).")

def calc_metrics_dict(model, X_te, y_te):
    pred = model.predict(X_te)
    return {
        'R² (Explication de la variance)': f"{r2_score(y_te, pred):.3f}",
        'MAE (Erreur Moyenne Absolue)': f"{mean_absolute_error(y_te, pred):.3f}",
        'RMSE (Erreur Quadratique)': f"{np.sqrt(mean_squared_error(y_te, pred)):.3f}"
    }

df_comparatif = pd.DataFrame([
    calc_metrics_dict(model_loud, X_test_l, y_test_l),
    calc_metrics_dict(model_aco, X_test_a, y_test_a),
    calc_metrics_dict(model_multi, X_test_m, y_test_m)
], index=["1. Simple (Loudness)", "2. Simple (Acousticness)", "3. Multiple (Toutes variables)"])

st.table(df_comparatif)

st.success("🏁 **Conclusion :** Le modèle multiple est le vainqueur incontesté. Avec un R² très supérieur et une erreur moyenne très faible, il prouve qu'un morceau de musique ne peut pas être résumé à un seul de ses aspects.")

st.divider()

# =========================================================
# 6. Simulateur interactif
# =========================================================
st.header("6. Testez le Modèle Multiple : Le Simulateur 🎛️")
st.markdown("Utilisez les curseurs pour définir les caractéristiques d'un morceau fictif. Le modèle exécutera son équation en temps réel pour en deviner l'énergie.")

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
    
    # Plafonnage entre 0 et 1 (règle mathématique d'affichage)
    prediction_bounded = np.clip(prediction, 0, 1)
    
    st.success(f"### ⚡ Énergie calculée par l'algorithme : {prediction_bounded:.2f} / 1.00")
    st.progress(float(prediction_bounded))