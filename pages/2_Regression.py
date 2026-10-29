# =========================================================
# Configuration des chemins d'importation
# =========================================================
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from config import RANDOM_STATE, TEST_SIZE, TARGET_REG
from data import load_data, clean_data

st.set_page_config(page_title="Régression ML", page_icon="🤖", layout="wide")

# =========================================================
# NAVIGATION PAR BARRE LATÉRALE (ROUTAGE)
# =========================================================
st.sidebar.title("Navigation Modèles")
page_choisie = st.sidebar.radio("Sélectionnez l'étude :", ["⚡ Prédire l'Énergie", "🔥 Prédire la Popularité (Bonus)"])

# Application de la DA (Couleurs) selon la page
if page_choisie == "⚡ Prédire l'Énergie":
    couleur_principale = "#1DB954" # Vert Spotify
    couleur_rgba = "rgba(29, 185, 84, 0.4)" # Vert transparent pour les tableaux
    titre_page = "🎧 Prédire l'énergie : Démarche et Modélisation"
else:
    couleur_principale = "#FF5722" # Orange Feu
    couleur_rgba = "rgba(255, 87, 34, 0.4)" # Orange transparent pour les tableaux
    titre_page = "🔥 Prédire la Popularité : Le grand défi (Bonus)"

st.markdown(f"""
    <style>
    .big-font {{
        font-size: 45px !important;
        font-weight: 800;
        color: {couleur_principale};
        text-align: center;
        margin-top: 10px;
        margin-bottom: 20px;
        text-shadow: 0px 2px 4px rgba(0,0,0,0.3);
    }}
    .stProgress > div > div > div > div {{
        background-color: {couleur_principale};
    }}
    div[data-testid="stMetricValue"] {{
        color: {couleur_principale};
    }}
    </style>
""", unsafe_allow_html=True)

# =========================================================
# FONCTIONS UTILITAIRES ET CHARGEMENTS EN CACHE
# =========================================================
def plot_scatter_custom(y_true, y_pred, title, color):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.3, color=color, ax=ax)
    if "Popularité" in titre_page:
        ax.plot([0, 100], [0, 100], color='red', linestyle='--', linewidth=2)
    else:
        ax.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Valeur Réelle")
    ax.set_ylabel("Valeur Prédite")
    return fig

def calc_metrics_dict(model, X_te, y_te):
    pred = model.predict(X_te)
    return {
        'R² (Explication variance)': r2_score(y_te, pred),
        'MAE (Erreur Moyenne Absolue)': mean_absolute_error(y_te, pred),
        'RMSE (Erreur Quadratique)': np.sqrt(mean_squared_error(y_te, pred))
    }

@st.cache_data(show_spinner=False)
def prepare_data_and_models():
    df = load_data()
    df_clean = clean_data(df)
    
    # Base commune filtrée
    df_base = df_clean[
        (df_clean['duration_min'] >= 2.3) & (df_clean['duration_min'] <= 5.0) & 
        (df_clean['loudness'] >= -20.0) & (df_clean['tempo'] > 0) & (df_clean['speechiness'] < 0.5)
    ].copy()

    # Définition des variables
    NUMERIC_FEATURES = ['danceability', 'speechiness', 'acousticness', 'instrumentalness', 
                        'liveness', 'valence', 'loudness', 'tempo', 'key', 'mode', 
                        'duration_min', 'release_year']
    
    df_encoded = pd.get_dummies(df_base, columns=['playlist_genre'], drop_first=False, dtype=int)
    GENRE_FEATURES = [col for col in df_encoded.columns if col.startswith('playlist_genre_')]
    FEATURES_ENERGY_AVEC = NUMERIC_FEATURES + GENRE_FEATURES
    FEATURES_POP = NUMERIC_FEATURES + GENRE_FEATURES

    # ---- PARTIE ÉNERGIE (Modèles Simples) ----
    X_tr_l, X_te_l, y_tr_l, y_te_l = train_test_split(df_base[['loudness']], df_base[TARGET_REG], test_size=TEST_SIZE, random_state=RANDOM_STATE)
    model_lr_loud = LinearRegression().fit(X_tr_l, y_tr_l)
    model_rf_loud = RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=100).fit(X_tr_l, y_tr_l)
    
    X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(df_base[['acousticness']], df_base[TARGET_REG], test_size=TEST_SIZE, random_state=RANDOM_STATE)
    model_lr_aco = LinearRegression().fit(X_tr_a, y_tr_a)
    model_rf_aco = RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=100).fit(X_tr_a, y_tr_a)
    
    # ---- PARTIE ÉNERGIE (Modèles Multiples : Sans et Avec Genre) ----
    X_tr_m_sans, X_te_m_sans, y_tr_m_sans, y_te_m_sans = train_test_split(df_base[NUMERIC_FEATURES], df_base[TARGET_REG], test_size=TEST_SIZE, random_state=RANDOM_STATE)
    model_lr_multi_sans = LinearRegression().fit(X_tr_m_sans, y_tr_m_sans)
    model_rf_multi_sans = RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=100).fit(X_tr_m_sans, y_tr_m_sans)

    X_tr_m_avec, X_te_m_avec, y_tr_m_avec, y_te_m_avec = train_test_split(df_encoded[FEATURES_ENERGY_AVEC], df_encoded[TARGET_REG], test_size=TEST_SIZE, random_state=RANDOM_STATE)
    model_lr_multi_avec = LinearRegression().fit(X_tr_m_avec, y_tr_m_avec)
    model_rf_multi_avec = RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=100).fit(X_tr_m_avec, y_tr_m_avec)

    # ---- PARTIE POPULARITÉ ----
    TARGET_POP = 'track_popularity'
    X_tr_p_l, X_te_p_l, y_tr_p_l, y_te_p_l = train_test_split(df_base[['loudness']], df_base[TARGET_POP], test_size=TEST_SIZE, random_state=RANDOM_STATE)
    model_pop_lr_loud = LinearRegression().fit(X_tr_p_l, y_tr_p_l)
    
    X_tr_p_d, X_te_p_d, y_tr_p_d, y_te_p_d = train_test_split(df_base[['danceability']], df_base[TARGET_POP], test_size=TEST_SIZE, random_state=RANDOM_STATE)
    model_pop_lr_dance = LinearRegression().fit(X_tr_p_d, y_tr_p_d)

    X_tr_p_m, X_te_p_m, y_tr_p_m, y_te_p_m = train_test_split(df_encoded[FEATURES_POP], df_encoded[TARGET_POP], test_size=TEST_SIZE, random_state=RANDOM_STATE)
    model_pop_lr_multi = LinearRegression().fit(X_tr_p_m, y_tr_p_m)
    model_pop_rf_multi = RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=100, max_depth=15).fit(X_tr_p_m, y_tr_p_m)

    return (df_clean, df_base, NUMERIC_FEATURES, GENRE_FEATURES, FEATURES_ENERGY_AVEC, FEATURES_POP,
            model_lr_loud, model_rf_loud, X_te_l, y_te_l,
            model_lr_aco, model_rf_aco, X_te_a, y_te_a,
            model_lr_multi_sans, model_rf_multi_sans, X_te_m_sans, y_te_m_sans,
            model_lr_multi_avec, model_rf_multi_avec, X_te_m_avec, y_te_m_avec,
            model_pop_lr_loud, X_te_p_l, y_te_p_l,
            model_pop_lr_dance, X_te_p_d, y_te_p_d,
            model_pop_lr_multi, model_pop_rf_multi, X_te_p_m, y_te_p_m)

with st.spinner("⏳ Entraînement en mémoire de l'environnement (sans fichiers joblib lourds)..."):
    (df_clean, df_base, NUMERIC_FEATURES, GENRE_FEATURES, FEATURES_ENERGY_AVEC, FEATURES_POP,
     model_lr_loud, model_rf_loud, X_test_l, y_test_l,
     model_lr_aco, model_rf_aco, X_test_a, y_test_a,
     model_lr_multi_sans, model_rf_multi_sans, X_test_m_sans, y_test_m_sans,
     model_lr_multi_avec, model_rf_multi_avec, X_test_m_avec, y_test_m_avec,
     model_pop_lr_loud, X_test_pop_l, y_test_pop_l,
     model_pop_lr_dance, X_test_pop_d, y_test_pop_d,
     model_pop_lr_multi, model_pop_rf_multi, X_test_pop_m, y_test_pop_m) = prepare_data_and_models()

# =========================================================
# CONTENU COMMUN : INTRODUCTION ET NETTOYAGE
# =========================================================
st.title(titre_page)

if page_choisie == "⚡ Prédire l'Énergie":
    st.info("🎯 **Notre défi :** Peut-on deviner mathématiquement l'énergie d'un morceau de musique sans jamais l'écouter, uniquement en analysant ses caractéristiques techniques ?")
else:
    st.info("🎯 **Notre défi :** Peut-on deviner mathématiquement le succès (Popularité de 0 à 100) d'un morceau en analysant ses caractéristiques audio ?")

st.header("1. Préparation des données et concept de Train/Test Split")

cond_dur = (df_clean['duration_min'] < 2.3) | (df_clean['duration_min'] > 5.0)
cond_loud = df_clean['loudness'] < -20.0
cond_tempo = df_clean['tempo'] == 0
cond_speech = df_clean['speechiness'] >= 0.5
df_analyse = pd.DataFrame({
    'Critère aberrant': ['Durée hors norme (<2.3m ou >5.0m)', 'Quasi-silence (Loudness < -20 dB)', 'Erreur API (Tempo = 0 BPM)', 'Podcasts (Speechiness >= 0.5)'],
    'Pistes impactées': [cond_dur.sum(), cond_loud.sum(), cond_tempo.sum(), cond_speech.sum()],
    'Proportion': [f"{(cond_dur.mean()*100):.2f} %", f"{(cond_loud.mean()*100):.2f} %", f"{(cond_tempo.mean()*100):.2f} %", f"{(cond_speech.mean()*100):.2f} %"],
    'Stratégie': ['Suppression', 'Suppression', 'Suppression', 'Suppression']
})
st.table(df_analyse)

with st.expander("💡 Débat Data Science : Pourquoi la Suppression plutôt que le Remplacement (Imputation) ?"):
    st.markdown("""
    En Data Science, on peut souvent remplacer une donnée aberrante par la médiane du dataset. Dans notre cas précis, cette approche serait une erreur :
    * **Remplacer la nature d'un objet :** Si nous remplaçons le `speechiness` d'un podcast (0.90) par la moyenne globale (0.05), le morceau reste intrinsèquement un podcast !
    * **Restriction stratégique :** Filtrer les durées entre 2.3 et 5.0 minutes est un choix assumé pour centrer notre algorithme sur le "format radio standard".
    """)

st.code("""
# Exemple de filtrage global pour nos modèles suite à l'analyse
df_multi = df_clean[
    (df_clean['duration_min'] >= 2.3) & (df_clean['duration_min'] <= 5.0) & # Format radio standard
    (df_clean['loudness'] >= -20.0) &                                       # Exclut les quasi-silences
    (df_clean['tempo'] > 0) &                                               # Exclut les erreurs d'API
    (df_clean['speechiness'] < 0.5)                                         # Exclut les podcasts (trop parlés)
]
""", language="python")

st.markdown("""
### La séparation Train / Test
Pour vérifier qu'un étudiant a compris, on lui donne un examen inédit. En Machine Learning, c'est la fonction `train_test_split`.
- **Train (80%)** : Le modèle observe les caractéristiques et l'énergie réelle, puis calcule ses règles.
- **Test (20%)** : Le modèle découvre de nouveaux morceaux. On lui cache l'énergie, il doit la prédire, et on mesure son erreur.
""")

st.code("""
# Découpage Train/Test avec une graine fixe (random_state=42) pour la reproductibilité
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
""", language="python")
st.divider()

# =========================================================
# ROUTAGE DES PAGES (ÉNERGIE VS POPULARITÉ)
# =========================================================

if page_choisie == "⚡ Prédire l'Énergie":
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Options des Modèles")
    inclure_genre_global = st.sidebar.checkbox("Inclure le genre musical (Énergie)", value=True)
    
    if inclure_genre_global:
        model_lr_multi_actif = model_lr_multi_avec
        model_rf_multi_actif = model_rf_multi_avec
        X_test_m_actif = X_test_m_avec
        y_test_m_actif = y_test_m_avec
        nom_modele = "Multiple (+ Genres)"
    else:
        model_lr_multi_actif = model_lr_multi_sans
        model_rf_multi_actif = model_rf_multi_sans
        X_test_m_actif = X_test_m_sans
        y_test_m_actif = y_test_m_sans
        nom_modele = "Multiple (Sans Genre)"

    tab_lr, tab_rf = st.tabs(["📈 Modèle 1 : Régression Linéaire", "🌲 Modèle 2 : Random Forest"])

    with tab_lr:
        st.header("2. Modèle Simple A : Prédiction par le Volume (`loudness`)")
        st.code("model_loud = LinearRegression().fit(X_train_l, y_train_l)", language="python")
        col_a1, col_a2 = st.columns([1, 2])
        y_pred_lr_l = model_lr_loud.predict(X_test_l)
        with col_a1:
            st.metric("Précision (R² Test)", f"{r2_score(y_test_l, y_pred_lr_l):.3f}")
            st.metric("Erreur (MAE Test)", f"{mean_absolute_error(y_test_l, y_pred_lr_l):.3f}")
        with col_a2:
            st.pyplot(plot_scatter_custom(y_test_l, y_pred_lr_l, "Modèle Linéaire A (Loudness)", couleur_principale))
        st.divider()

        st.header("3. Modèle Simple B : Prédiction par l'Acoustique (`acousticness`)")
        st.code("model_aco = LinearRegression().fit(X_train_a, y_train_a)", language="python")
        col_b1, col_b2 = st.columns([1, 2])
        y_pred_lr_a = model_lr_aco.predict(X_test_a)
        with col_b1:
            st.metric("Précision (R² Test)", f"{r2_score(y_test_a, y_pred_lr_a):.3f}")
            st.metric("Erreur (MAE Test)", f"{mean_absolute_error(y_test_a, y_pred_lr_a):.3f}")
        with col_b2:
            st.pyplot(plot_scatter_custom(y_test_a, y_pred_lr_a, "Modèle Linéaire B (Acousticness)", couleur_principale))
        st.divider()

        st.header(f"4. Modèle {nom_modele} : La puissance de toutes les variables")
        st.code("model_multi = LinearRegression().fit(X_train_m, y_train_m)", language="python")
        col_m1, col_m2 = st.columns([1, 2])
        y_pred_lr_m = model_lr_multi_actif.predict(X_test_m_actif)
        with col_m1:
            st.metric("⭐ Précision (R² Test)", f"{r2_score(y_test_m_actif, y_pred_lr_m):.3f}")
            st.metric("⭐ Erreur (MAE Test)", f"{mean_absolute_error(y_test_m_actif, y_pred_lr_m):.3f}")
        with col_m2:
            st.pyplot(plot_scatter_custom(y_test_m_actif, y_pred_lr_m, f"Modèle Linéaire {nom_modele}", couleur_principale))

        st.subheader("Bilan Comparatif (Linéaire)")
        df_comparatif_lr = pd.DataFrame([calc_metrics_dict(model_lr_loud, X_test_l, y_test_l), calc_metrics_dict(model_lr_aco, X_test_a, y_test_a), calc_metrics_dict(model_lr_multi_actif, X_test_m_actif, y_test_m_actif)], index=["1. Simple (Loudness)", "2. Simple (Acousticness)", f"3. {nom_modele}"])
        
        styled_df_lr = df_comparatif_lr.style\
            .format("{:.3f}")\
            .highlight_max(subset=['R² (Explication variance)'], color=couleur_rgba)\
            .highlight_min(subset=['MAE (Erreur Moyenne Absolue)', 'RMSE (Erreur Quadratique)'], color=couleur_rgba)
        st.dataframe(styled_df_lr, use_container_width=True)
        st.divider()

        st.header("5. Testez la Régression Linéaire 🎛️")
        
        genre_choisi_lr = None
        if inclure_genre_global:
            genre_choisi_lr = st.selectbox("🎸 Genre Musical :", options=['pop', 'rap', 'rock', 'latin', 'r&b', 'edm'], key="genre_lr")

        with st.form("form_lr", border=True):
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                loudness_lr = st.slider("🔊 Loudness (dB)", -30.0, 5.0, -5.0, 0.5, key="el_loud")
                acousticness_lr = st.slider("🎸 Acousticness", 0.0, 1.0, 0.2, 0.05, key="el_aco")
                danceability_lr = st.slider("🕺 Danceability", 0.0, 1.0, 0.6, 0.05, key="el_dance")
                tempo_lr = st.number_input("⏱️ Tempo (BPM)", 0.0, 250.0, 120.0, key="el_tempo")
            with col_s2:
                valence_lr = st.slider("😊 Valence", 0.0, 1.0, 0.5, 0.05, key="el_val")
                instrumentalness_lr = st.slider("🎻 Instrumentalness", 0.0, 1.0, 0.0, 0.05, key="el_instru")
                liveness_lr = st.slider("🎤 Liveness", 0.0, 1.0, 0.15, 0.05, key="el_live")
                speechiness_lr = st.slider("🗣️ Speechiness", 0.0, 1.0, 0.05, 0.05, key="el_speech")
            with col_s3:
                key_lr = st.selectbox("🎹 Key", options=list(range(12)), key="el_key")
                mode_lr = st.selectbox("🎼 Mode", options=[0, 1], key="el_mode")
                duration_lr = st.number_input("⏳ Durée (min)", 0.5, 15.0, 3.5, 0.1, key="el_dur")
                release_yr_lr = st.number_input("📅 Année", 1950, 2024, 2020, 1, key="el_yr")
            sub_lr = st.form_submit_button("⚡ PRÉDIRE (LINÉAIRE)", use_container_width=True)

        if sub_lr:
            input_dict = {'danceability': danceability_lr, 'speechiness': speechiness_lr, 'acousticness': acousticness_lr, 'instrumentalness': instrumentalness_lr, 'liveness': liveness_lr, 'valence': valence_lr, 'loudness': loudness_lr, 'tempo': tempo_lr, 'key': key_lr, 'mode': mode_lr, 'duration_min': duration_lr, 'release_year': release_yr_lr}
            
            if inclure_genre_global and genre_choisi_lr:
                for col in GENRE_FEATURES:
                    input_dict[col] = 1 if col == f'playlist_genre_{genre_choisi_lr}' else 0
                in_lr = pd.DataFrame([input_dict])[FEATURES_ENERGY_AVEC]
            else:
                in_lr = pd.DataFrame([input_dict])[NUMERIC_FEATURES]

            pred_lr = np.clip(model_lr_multi_actif.predict(in_lr)[0], 0, 1)
            st.markdown(f"<div class='big-font'>⚡ ÉNERGIE CALCULÉE : {pred_lr:.2f} / 1.00</div>", unsafe_allow_html=True)
            st.progress(float(pred_lr))

    with tab_rf:
        st.header("2. RF Simple A : Par le Volume (`loudness`)")
        st.code("model_rf_loud = RandomForestRegressor(random_state=42).fit(X_train_l, y_train_l)", language="python")
        col_rfa1, col_rfa2 = st.columns([1, 2])
        y_pred_rf_l = model_rf_loud.predict(X_test_l)
        with col_rfa1:
            st.metric("Précision (R² Test)", f"{r2_score(y_test_l, y_pred_rf_l):.3f}")
            st.markdown("**Observation :** Regardez les points ! L'arbre de décision crée des \"paliers\".")
        with col_rfa2:
            st.pyplot(plot_scatter_custom(y_test_l, y_pred_rf_l, "RF Simple A (Loudness)", couleur_principale))
        st.divider()
        
        st.header(f"3. RF {nom_modele} : La vraie puissance des Forêts Aléatoires")
        st.code("model_rf_multi = RandomForestRegressor(random_state=42).fit(X_train_m, y_train_m)", language="python")
        col_rfm1, col_rfm2 = st.columns([1, 2])
        y_pred_rf_m = model_rf_multi_actif.predict(X_test_m_actif)
        with col_rfm1:
            st.metric("🏆 Précision (R² Test)", f"{r2_score(y_test_m_actif, y_pred_rf_m):.3f}")
            st.metric("📉 Erreur (MAE Test)", f"{mean_absolute_error(y_test_m_actif, y_pred_rf_m):.3f}")
        with col_rfm2:
            st.pyplot(plot_scatter_custom(y_test_m_actif, y_pred_rf_m, f"Random Forest {nom_modele}", couleur_principale))

        st.subheader("Bilan Comparatif Final (Random Forest)")
        df_comparatif_rf = pd.DataFrame([calc_metrics_dict(model_rf_loud, X_test_l, y_test_l), calc_metrics_dict(model_rf_aco, X_test_a, y_test_a), calc_metrics_dict(model_rf_multi_actif, X_test_m_actif, y_test_m_actif)], index=["1. RF Simple (Loudness)", "2. RF Simple (Acousticness)", f"3. RF {nom_modele}"])
        
        styled_df_rf = df_comparatif_rf.style\
            .format("{:.3f}")\
            .highlight_max(subset=['R² (Explication variance)'], color=couleur_rgba)\
            .highlight_min(subset=['MAE (Erreur Moyenne Absolue)', 'RMSE (Erreur Quadratique)'], color=couleur_rgba)
        st.dataframe(styled_df_rf, use_container_width=True)
        st.divider()

        st.header("4. Testez le Random Forest 🎛️")
        
        genre_choisi_rf = None
        if inclure_genre_global:
            genre_choisi_rf = st.selectbox("🎸 Genre Musical :", options=['pop', 'rap', 'rock', 'latin', 'r&b', 'edm'], key="genre_rf")

        with st.form("form_rf", border=True):
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                loudness_rf = st.slider("🔊 Loudness (dB)", -30.0, 5.0, -5.0, 0.5, key="er_loud")
                acousticness_rf = st.slider("🎸 Acousticness", 0.0, 1.0, 0.2, 0.05, key="er_aco")
                danceability_rf = st.slider("🕺 Danceability", 0.0, 1.0, 0.6, 0.05, key="er_dance")
                tempo_rf = st.number_input("⏱️ Tempo (BPM)", 0.0, 250.0, 120.0, key="er_tempo")
            with col_r2:
                valence_rf = st.slider("😊 Valence", 0.0, 1.0, 0.5, 0.05, key="er_val")
                instrumentalness_rf = st.slider("🎻 Instrumentalness", 0.0, 1.0, 0.0, 0.05, key="er_instru")
                liveness_rf = st.slider("🎤 Liveness", 0.0, 1.0, 0.15, 0.05, key="er_live")
                speechiness_rf = st.slider("🗣️ Speechiness", 0.0, 1.0, 0.05, 0.05, key="er_speech")
            with col_r3:
                key_rf = st.selectbox("🎹 Key", options=list(range(12)), key="er_key")
                mode_rf = st.selectbox("🎼 Mode", options=[0, 1], key="er_mode")
                duration_rf = st.number_input("⏳ Durée (min)", 0.5, 15.0, 3.5, 0.1, key="er_dur")
                release_yr_rf = st.number_input("📅 Année", 1950, 2024, 2020, 1, key="er_yr")
            sub_rf = st.form_submit_button("⚡ PRÉDIRE (RANDOM FOREST)", use_container_width=True)

        if sub_rf:
            input_dict = {'danceability': danceability_rf, 'speechiness': speechiness_rf, 'acousticness': acousticness_rf, 'instrumentalness': instrumentalness_rf, 'liveness': liveness_rf, 'valence': valence_rf, 'loudness': loudness_rf, 'tempo': tempo_rf, 'key': key_rf, 'mode': mode_rf, 'duration_min': duration_rf, 'release_year': release_yr_rf}
            
            if inclure_genre_global and genre_choisi_rf:
                for col in GENRE_FEATURES:
                    input_dict[col] = 1 if col == f'playlist_genre_{genre_choisi_rf}' else 0
                in_rf = pd.DataFrame([input_dict])[FEATURES_ENERGY_AVEC]
            else:
                in_rf = pd.DataFrame([input_dict])[NUMERIC_FEATURES]

            pred_rf = np.clip(model_rf_multi_actif.predict(in_rf)[0], 0, 1)
            st.markdown(f"<div class='big-font'>⚡ ÉNERGIE CALCULÉE : {pred_rf:.2f} / 1.00</div>", unsafe_allow_html=True)
            st.progress(float(pred_rf))

elif page_choisie == "🔥 Prédire la Popularité (Bonus)":
    
    st.header("2. 🧠 Réponse à la question Bonus de l'énoncé")
    st.warning("⚠️ **Alerte Spoilers : Les scores de nos algorithmes vont s'effondrer.**")
    st.markdown("""
    Contrairement à l'Énergie où nos algorithmes excellaient, prédire la popularité avec strictement les mêmes caractéristiques audios donne de mauvais résultats. **Ceci n'est pas une erreur de code, c'est une conclusion sociologique majeure de notre étude :**
    1. **L'audio ne fait pas tout :** Un morceau mathématiquement "parfait" pour danser peut faire un flop si l'artiste est inconnu.
    2. **La notoriété et le marketing :** Le nom de l'artiste (non inclus ici) pèse souvent plus lourd que le tempo de la musique.
    3. **Le facteur temporel (Hype) :** Une musique peut exploser 10 ans après sa sortie grâce à TikTok. L'algorithme ne peut pas le deviner.
    """)
    st.divider()

    tab_pop_lr, tab_pop_rf = st.tabs(["📈 Modèle 1 : Régression Linéaire", "🌲 Modèle 2 : Random Forest"])

    with tab_pop_lr:
        st.header("3. Linéaire Multiple : Audio complet")
        st.code("model_pop_lr_multi = LinearRegression().fit(X_train_pop_m, y_train_pop_m)", language="python")
        col_m1, col_m2 = st.columns([1, 2])
        y_pred_pop_lr_m = model_pop_lr_multi.predict(X_test_pop_m)
        with col_m1:
            st.metric("⭐ Précision (R²)", f"{r2_score(y_test_pop_m, y_pred_pop_lr_m):.3f}")
            st.metric("⭐ Erreur (MAE)", f"{mean_absolute_error(y_test_pop_m, y_pred_pop_lr_m):.1f} pts")
        with col_m2:
            st.pyplot(plot_scatter_custom(y_test_pop_m, y_pred_pop_lr_m, "LR Multiple (Audio)", couleur_principale))

        st.subheader("Bilan Linéaire")
        df_comparatif_pop_lr = pd.DataFrame([calc_metrics_dict(model_pop_lr_loud, X_test_pop_l, y_test_pop_l), calc_metrics_dict(model_pop_lr_dance, X_test_pop_d, y_test_pop_d), calc_metrics_dict(model_pop_lr_multi, X_test_pop_m, y_test_pop_m)], index=["1. Simple (Loudness)", "2. Simple (Danceability)", "3. Multiple (Audio)"])
        
        styled_df_pop_lr = df_comparatif_pop_lr.style\
            .format("{:.3f}")\
            .highlight_max(subset=['R² (Explication variance)'], color=couleur_rgba)\
            .highlight_min(subset=['MAE (Erreur Moyenne Absolue)', 'RMSE (Erreur Quadratique)'], color=couleur_rgba)
        st.dataframe(styled_df_pop_lr, use_container_width=True)
        st.divider()

        st.header("4. Testez la Machine à Succès (Linéaire) 🎛️")
        
        genre_choisi_plr = st.selectbox("🎸 Genre Musical :", options=['pop', 'rap', 'rock', 'latin', 'r&b', 'edm'], key="genre_plr")

        with st.form("form_pop_lr", border=True):
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                loudness_lr = st.slider("🔊 Loudness (dB)", -30.0, 5.0, -5.0, 0.5, key="plr_loud")
                acousticness_lr = st.slider("🎸 Acousticness", 0.0, 1.0, 0.2, 0.05, key="plr_aco")
                danceability_lr = st.slider("🕺 Danceability", 0.0, 1.0, 0.6, 0.05, key="plr_dance")
                tempo_lr = st.number_input("⏱️ Tempo (BPM)", 0.0, 250.0, 120.0, key="plr_tempo")
            with col_s2:
                valence_lr = st.slider("😊 Valence", 0.0, 1.0, 0.5, 0.05, key="plr_val")
                instrumentalness_lr = st.slider("🎻 Instrumentalness", 0.0, 1.0, 0.0, 0.05, key="plr_instru")
                liveness_lr = st.slider("🎤 Liveness", 0.0, 1.0, 0.15, 0.05, key="plr_live")
                speechiness_lr = st.slider("🗣️ Speechiness", 0.0, 1.0, 0.05, 0.05, key="plr_speech")
            with col_s3:
                key_lr = st.selectbox("🎹 Key", options=list(range(12)), key="plr_key")
                mode_lr = st.selectbox("🎼 Mode", options=[0, 1], key="plr_mode")
                duration_lr = st.number_input("⏳ Durée (min)", 0.5, 15.0, 3.5, 0.1, key="plr_dur")
                release_yr_lr = st.number_input("📅 Année", 1950, 2024, 2020, 1, key="plr_yr")
            sub_lr = st.form_submit_button("🔥 PRÉDIRE LA POPULARITÉ (LINÉAIRE)", use_container_width=True)

        if sub_lr:
            input_dict = {'danceability': danceability_lr, 'speechiness': speechiness_lr, 'acousticness': acousticness_lr, 'instrumentalness': instrumentalness_lr, 'liveness': liveness_lr, 'valence': valence_lr, 'loudness': loudness_lr, 'tempo': tempo_lr, 'key': key_lr, 'mode': mode_lr, 'duration_min': duration_lr, 'release_year': release_yr_lr}
            for col in GENRE_FEATURES:
                input_dict[col] = 1 if col == f'playlist_genre_{genre_choisi_plr}' else 0

            in_lr = pd.DataFrame([input_dict])[FEATURES_POP]
            pred_lr = np.clip(model_pop_lr_multi.predict(in_lr)[0], 0, 100)
            st.markdown(f"<div class='big-font'>🔥 SCORE DE SUCCÈS : {pred_lr:.1f} / 100</div>", unsafe_allow_html=True)
            st.progress(int(pred_lr))

    with tab_pop_rf:
        st.header("3. Random Forest Multiple : La limite de la machine")
        st.code("model_pop_rf_multi = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=15).fit(X_train_pop_m, y_train_pop_m)", language="python")
        col_rfm1, col_rfm2 = st.columns([1, 2])
        y_pred_pop_rf_m = model_pop_rf_multi.predict(X_test_pop_m)
        with col_rfm1:
            st.metric("🏆 Précision (R²)", f"{r2_score(y_test_pop_m, y_pred_pop_rf_m):.3f}")
            st.metric("📉 Erreur (MAE)", f"{mean_absolute_error(y_test_pop_m, y_pred_pop_rf_m):.1f} pts")
        with col_rfm2:
            st.pyplot(plot_scatter_custom(y_test_pop_m, y_pred_pop_rf_m, "Random Forest (Audio)", couleur_principale))

        st.subheader("Bilan Comparatif Final (Popularité)")
        df_comparatif_pop_rf = pd.DataFrame([calc_metrics_dict(model_pop_lr_multi, X_test_pop_m, y_test_pop_m), calc_metrics_dict(model_pop_rf_multi, X_test_pop_m, y_test_pop_m)], index=["1. Linéaire Multiple (Audio)", "2. RF Multiple (Audio)"])
        
        styled_df_pop_rf = df_comparatif_pop_rf.style\
            .format("{:.3f}")\
            .highlight_max(subset=['R² (Explication variance)'], color=couleur_rgba)\
            .highlight_min(subset=['MAE (Erreur Moyenne Absolue)', 'RMSE (Erreur Quadratique)'], color=couleur_rgba)
        st.dataframe(styled_df_pop_rf, use_container_width=True)
        st.divider()

        st.header("4. Testez la Machine à Succès (Random Forest) 🎛️")
        
        genre_choisi_prf = st.selectbox("🎸 Genre Musical :", options=['pop', 'rap', 'rock', 'latin', 'r&b', 'edm'], key="genre_prf")

        with st.form("form_pop_rf", border=True):
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                loudness_rf = st.slider("🔊 Loudness (dB)", -30.0, 5.0, -5.0, 0.5, key="prf_loud")
                acousticness_rf = st.slider("🎸 Acousticness", 0.0, 1.0, 0.2, 0.05, key="prf_aco")
                danceability_rf = st.slider("🕺 Danceability", 0.0, 1.0, 0.6, 0.05, key="prf_dance")
                tempo_rf = st.number_input("⏱️ Tempo (BPM)", 0.0, 250.0, 120.0, key="prf_tempo")
            with col_r2:
                valence_rf = st.slider("😊 Valence", 0.0, 1.0, 0.5, 0.05, key="prf_val")
                instrumentalness_rf = st.slider("🎻 Instrumentalness", 0.0, 1.0, 0.0, 0.05, key="prf_instru")
                liveness_rf = st.slider("🎤 Liveness", 0.0, 1.0, 0.15, 0.05, key="prf_live")
                speechiness_rf = st.slider("🗣️ Speechiness", 0.0, 1.0, 0.05, 0.05, key="prf_speech")
            with col_r3:
                key_rf = st.selectbox("🎹 Key", options=list(range(12)), key="prf_key")
                mode_rf = st.selectbox("🎼 Mode", options=[0, 1], key="prf_mode")
                duration_rf = st.number_input("⏳ Durée (min)", 0.5, 15.0, 3.5, 0.1, key="prf_dur")
                release_yr_rf = st.number_input("📅 Année", 1950, 2024, 2020, 1, key="prf_yr")
            sub_rf = st.form_submit_button("🔥 PRÉDIRE LA POPULARITÉ (RANDOM FOREST)", use_container_width=True)

        if sub_rf:
            input_dict = {'danceability': danceability_rf, 'speechiness': speechiness_rf, 'acousticness': acousticness_rf, 'instrumentalness': instrumentalness_rf, 'liveness': liveness_rf, 'valence': valence_rf, 'loudness': loudness_rf, 'tempo': tempo_rf, 'key': key_rf, 'mode': mode_rf, 'duration_min': duration_rf, 'release_year': release_yr_rf}
            for col in GENRE_FEATURES:
                input_dict[col] = 1 if col == f'playlist_genre_{genre_choisi_prf}' else 0

            in_rf = pd.DataFrame([input_dict])[FEATURES_POP]
            pred_rf = np.clip(model_pop_rf_multi.predict(in_rf)[0], 0, 100)
            st.markdown(f"<div class='big-font'>🔥 SCORE DE SUCCÈS : {pred_rf:.1f} / 100</div>", unsafe_allow_html=True)