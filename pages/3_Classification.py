# =========================================================
# Configuration des chemins d'importation
# =========================================================
import sys
import os
import time

# Permet à Python de trouver les fichiers config.py et data.py à la racine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

# Import des configurations communes
from config import RANDOM_STATE, TEST_SIZE, TARGET_CLF, FEATURES_CLF
from data import load_data, clean_data

st.set_page_config(page_title="Classification - Genre", page_icon="🎵", layout="wide")

# =========================================================
# CSS Personnalisé (Thème Spotify DA)
# =========================================================
# --- BLOC DA SPOTIFY À METTRE AU DÉBUT DE CHAQUE PAGE ---
LOGO_ICON = "https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg"
LOGO_FULL = "https://upload.wikimedia.org/wikipedia/commons/2/26/Spotify_logo_with_text.svg"

# Logo dans l'onglet du navigateur
st.set_page_config(page_title=" Prédire le genre musical : Démarche et Modélisation", page_icon=LOGO_ICON, layout="wide")

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
    [data-testid="stHeader"] {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

# Titre avec le logo Spotify à côté
st.markdown(f"""
<div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
    <img src="{LOGO_ICON}" width="50">
    <h1 style="margin: 0; font-size: 40px;"> Prédire le genre musical : Démarche et Modélisation</h1>
</div>
""", unsafe_allow_html=True)
# ---------------------------------------------------------



st.info("🎯 **Notre défi :** L'algorithme peut-il classer automatiquement un morceau dans l'un des 6 grands genres de la plateforme, uniquement à partir de ses données techniques ?")

st.markdown("""
Cette page présente notre modèle de **Classification** conçu pour prédire la variable cible **`playlist_genre`**.
Les genres possibles sont : `edm`, `latin`, `pop`, `r&b`, `rap`, et `rock`.

**Pour concevoir un modèle robuste, notre démarche suit 3 étapes :**
1. 🧮 **Préparation et Split Stratifié** pour garantir que chaque genre soit équitablement représenté.
2. 📊 **Comparaison d'algorithmes** (Régression Logistique, KNN, Random Forest) pour retenir le plus performant.
3. 📉 **Analyse Fine (Matrice de Confusion)** pour comprendre quels genres l'IA confond naturellement.
""")
st.divider()

# =========================================================
# Chargement des données et préparation en cache
# =========================================================
@st.cache_data(show_spinner=False)
def prepare_classification_data():
    df = load_data()
    df_clean = clean_data(df)
    
    features_finales = FEATURES_CLF + ['track_popularity']
    
    X = df_clean[features_finales]
    y = df_clean[TARGET_CLF]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    return df_clean, X_train, X_test, y_train, y_test

with st.spinner("⏳ Chargement du catalogue et préparation du fractionnement..."):
    df_clf, X_train_c, X_test_c, y_train_c, y_test_c = prepare_classification_data()

@st.cache_resource
def load_classification_model():
    racine_projet = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    chemin_modele = os.path.join(racine_projet, 'models', 'classification.joblib')
    return joblib.load(chemin_modele)

try:
    model_clf = load_classification_model()
except FileNotFoundError:
    st.error("Modèle introuvable. Exécutez le notebook 03_classification.ipynb pour générer 'models/classification.joblib'.")
    st.stop()

# Génération des prédictions sur le test set
y_pred_c = model_clf.predict(X_test_c)


# =========================================================
# 1. Comparaison des modèles testés
# =========================================================
st.header("1. 🧹 Démarche et Comparaison des modèles")

st.markdown("""
Avant de choisir notre algorithme final, nous avons défini un **score plancher (Modèle Naïf)** : un algorithme qui répondrait totalement au hasard ou qui prédirait toujours la classe majoritaire obtiendrait environ **16.67 %** à **18 %** de réussite. 
Nous avons ensuite entraîné et comparé 3 algorithmes de Machine Learning :
""")

# Tableau comparatif statique (issu des résultats du notebook)
df_comparatif = pd.DataFrame({
    "Modèle": ["Modèle Naïf (Hasard/Majorité)", "Régression Logistique", "K-Nearest Neighbors (k=21)", "Random Forest"],
    "Accuracy": ["~ 18.5 %", "~ 50.7 %", "~ 53.6 %", "~ 60.2 %"],
    "F1-Score (Macro)": ["N/A", "~ 49.6 %", "N/A", "~ 59.3 %"]
})
st.dataframe(df_comparatif, use_container_width=True, hide_index=True)

st.success("🏁 **Choix final :** Le **Random Forest** a été retenu. En combinant 100 arbres de décision, il capture bien mieux les relations complexes entre les variables audio que les modèles linéaires.")
st.divider()


# =========================================================
# 2. Performances Globales (Random Forest)
# =========================================================
st.header("2. 🏆 Performances du modèle Random Forest")

col_c1, col_c2, col_c3, col_c4 = st.columns(4)
col_c1.metric("Accuracy Globale", f"{accuracy_score(y_test_c, y_pred_c):.3f}")
col_c2.metric("Précision (Macro)", f"{precision_score(y_test_c, y_pred_c, average='macro'):.3f}")
col_c3.metric("Rappel (Macro)", f"{recall_score(y_test_c, y_pred_c, average='macro'):.3f}")
col_c4.metric("F1-Score (Macro)", f"{f1_score(y_test_c, y_pred_c, average='macro'):.3f}")

with st.expander("🧐 Le Plafond de verre : Pourquoi stagnons-nous autour de 60 % ?"):
    st.markdown("""
    Après de très nombreux tests et un nettoyage approfondi du DataFrame (suppression des doublons via `track_id`, traitement minutieux des types, ajout de variables saines et corrélées comme `track_popularity`), **les performances se stabilisent naturellement autour de 60 %.**
    
    Ce résultat n'est pas un bug dans le code ou un manque d'optimisation. Il représente une limite inhérente à la nature même de la musique : le **chevauchement acoustique naturel**. 
    De nombreux genres musicaux (comme la Pop, le R&B, et la musique Latin) partagent des profils audio et des rythmiques très proches. Une musique Pop peut être très inspirée R&B, rendant la frontière floue même pour une oreille humaine.
    """)

st.divider()


# =========================================================
# 3. Analyse Approfondie : Matrice de Confusion
# =========================================================
st.header("3. 📉 Où le modèle se trompe-t-il ? (Matrice de confusion)")
st.markdown("La matrice de confusion détaille précisément comment le modèle classe chaque genre. La diagonale représente les bonnes prédictions.")

fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test_c, y_pred_c, labels=model_clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_clf.classes_)
disp.plot(cmap=plt.cm.Blues, ax=ax_cm, xticks_rotation=45)
ax_cm.set_title("Matrice de Confusion (Données de Test)", fontweight='bold')
st.pyplot(fig_cm)

st.markdown("""
**Interprétation Musicale :**
* 🎸 **Le Rock et l'EDM** sont les genres les plus faciles à reconnaître pour l'algorithme. Leurs signatures acoustiques (tempo rapide et régulier pour l'EDM, forte instrumentalité et énergie pour le Rock) sont très claires.
* 🎤 **La Pop** est la classe la plus souvent confondue par notre modèle. Cela est totalement cohérent d'un point de vue musical : la Pop est un genre "carrefour" qui se nourrit des codes de l'EDM, du R&B ou du Rap pour toucher un large public.
""")
st.divider()


# =========================================================
# 4. Simulateur Interactif & Analyseur de Catalogue
# =========================================================
st.header("4. 🎛️ Testez le Modèle : L'Analyseur de Genre")
st.markdown("Jouez avec les caractéristiques d'un morceau fictif, ou choisissez un vrai titre du catalogue pour voir comment l'algorithme répartit ses probabilités entre les 6 genres.")

# Ajout d'un troisième onglet pour l'animation HTML
tab_manual, tab_catalog, tab_labo = st.tabs(["🎚️ Créer un morceau manuellement", "🔍 Analyser un titre du catalogue", "🧪 Labo : Trois classifieurs en mouvement"])

# Onglet A : Saisie manuelle
with tab_manual:
    with st.form("clf_manual_form", border=True):
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            danceability_m = st.slider("🕺 Danceability", 0.0, 1.0, 0.5, 0.05, key="m_dan")
            energy_m = st.slider("⚡ Energy", 0.0, 1.0, 0.7, 0.05, key="m_nrj")
            speechiness_m = st.slider("🗣️ Speechiness", 0.0, 1.0, 0.1, 0.05, key="m_spe")
            acousticness_m = st.slider("🎸 Acousticness", 0.0, 1.0, 0.5, 0.05, key="m_aco")
            instrumentalness_m = st.slider("🎻 Instrumentalness", 0.0, 1.0, 0.0, 0.05, key="m_ins")
            
        with col_m2:
            liveness_m = st.slider("🎤 Liveness", 0.0, 1.0, 0.2, 0.05, key="m_liv")
            valence_m = st.slider("😊 Valence", 0.0, 1.0, 0.5, 0.05, key="m_val")
            loudness_m = st.slider("🔊 Loudness (dB)", -60.0, 0.0, -5.0, 0.5, key="m_lou")
            tempo_m = st.number_input("⏱️ Tempo (BPM)", 0.0, 250.0, 120.0, key="m_tem")
            
        with col_m3:
            key_m = st.selectbox("🎹 Key", options=list(range(12)), key="m_key")
            mode_m = st.selectbox("🎼 Mode", options=[0, 1], key="m_mod")
            duration_min_m = st.number_input("⏳ Durée (min)", 0.5, 20.0, 3.5, key="m_dur")
            release_year_m = st.number_input("📅 Année", 1950, 2026, 2020, key="m_yea")
            popularity_m = st.slider("⭐ Popularité", 0, 100, 50, key="m_pop")

        submit_manual = st.form_submit_button("🔮 CLASSIFIER CE MORCEAU FICTIF", use_container_width=True)

# Onglet B : Catalogue existant
with tab_catalog:
    titres_disponibles = df_clf['track_name'].dropna().unique()
    titre_choisi = st.selectbox("Rechercher un morceau dans la base Spotify :", titres_disponibles)
    submit_auto = st.button("🔮 ANALYSER CE TITRE RÉEL", use_container_width=True, type="primary")

# Onglet C : Fichier HTML interactif
with tab_labo:
    st.markdown("Cette animation interactive compare le comportement de la Régression Logistique, du KNN et du Random Forest sur un même jeu de données.")
    
    # Chargement et affichage du fichier HTML
    racine_projet = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    chemin_html = os.path.join(racine_projet, 'class_mouv.html')
    
    try:
        with open(chemin_html, 'r', encoding='utf-8') as f:
            html_data = f.read()
        components.html(html_data, height=1200, scrolling=True)
    except FileNotFoundError:
        st.warning("Le fichier d'animation HTML (class_mouv.html) n'a pas été trouvé à la racine du projet.")

# Exécution de la prédiction (seulement si le bouton manuel ou auto est cliqué)
if submit_manual or submit_auto:
    with st.spinner("Analyse des vecteurs audio en cours..."):
        time.sleep(0.8) # Effet visuel
        
        features_finales = FEATURES_CLF + ['track_popularity']
        
        if submit_auto:
            # Récupération de la première occurrence du morceau sélectionné
            morceau_data = df_clf[df_clf['track_name'] == titre_choisi].iloc[0]
            genre_reel = morceau_data['playlist_genre'].upper()
            st.info(f"**Vrai genre attribué par Spotify :** {genre_reel}")
            
            # Préparation du dataframe d'entrée avec les bonnes colonnes
            user_data = pd.DataFrame([morceau_data[features_finales]])
        else:
            # Construction du dataframe d'après le formulaire manuel
            user_input = {
                'danceability': [danceability_m], 'energy': [energy_m], 'speechiness': [speechiness_m],
                'acousticness': [acousticness_m], 'instrumentalness': [instrumentalness_m],
                'liveness': [liveness_m], 'valence': [valence_m], 'loudness': [loudness_m],
                'tempo': [tempo_m], 'key': [key_m], 'mode': [mode_m],
                'duration_min': [duration_min_m], 'release_year': [release_year_m],
                'track_popularity': [popularity_m]
            }
            user_data = pd.DataFrame(user_input)[features_finales]

        # Utilisation du modèle entraîné par l'équipe
        probas = model_clf.predict_proba(user_data)[0]
        classes = model_clf.classes_
        
        # Identification du genre gagnant
        genre_predit = classes[np.argmax(probas)]
        
        # Formatage des probabilités pour le graphique (en %)
        df_probas = pd.DataFrame({"Probabilité (%)": probas * 100}, index=classes)
        
        st.markdown(f"<div class='big-font'>🎵 GENRE PRÉDIT : {genre_predit.upper()}</div>", unsafe_allow_html=True)
        
        st.subheader("Répartition de la certitude du modèle")
        st.bar_chart(df_probas, color="#1DB954")