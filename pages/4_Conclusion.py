import streamlit as st

st.set_page_config(page_title="Conclusion", page_icon="🏁", layout="wide")

st.title("🏁 Bilan et Conclusion")
st.markdown("Cette page synthétise les apprentissages de notre projet et apporte une réponse à la start-up musicale.")

# --- 1. RÉPONSES AUX PROBLÉMATIQUES ---
st.header("1. Réponses aux problématiques")
st.markdown("""
*(À remplir par l'équipe après analyse finale)*

* **Prédire l'énergie (Régression) :** Est-ce possible avec précision ? Quelles sont les variables qui influencent le plus le dynamisme d'un morceau (ex: Loudness, Acousticness) ?
* **Prédire le genre (Classification) :** Les frontières entre les genres musicaux sont-elles nettes ou floues ? Quels genres le modèle confond-il le plus souvent (ex: Pop vs EDM) et pourquoi ?
""")

# --- 2. LIMITES DES MODÈLES ---
st.header("2. Limites identifiées")
col1, col2 = st.columns(2)

with col1:
    st.info("📉 **Limites Techniques**")
    st.markdown("""
    * **Overfitting (Surapprentissage) :** *(Ajouter ici si le KNN ou le Random Forest a mémorisé les données).*
    * **Déséquilibre ou subjectivité :** L'assignation d'un morceau à une playlist par un humain est parfois arbitraire (un morceau peut être à la fois Pop et Dance).
    """)

with col2:
    st.info("🎵 **Limites Musicales**")
    st.markdown("""
    * **Variables limitées :** Les indicateurs Spotify (danceability, valence) sont des boîtes noires déjà pré-calculées. 
    * **Absence des paroles (Lyrics) :** Le genre "Rap" pourrait être bien mieux détecté si l'on analysait le texte en plus de l'audio.
    """)

# --- 3. PISTES D'AMÉLIORATION ---
st.header("3. Pistes d'amélioration")
st.markdown("""
Pour aller plus loin et fournir un outil plus robuste à la start-up, nous pourrions :
1. **Intégrer le NLP (Traitement du Langage Naturel) :** Analyser les paroles des musiques pour aider la classification.
2. **Traitement du signal audio brut :** Utiliser des réseaux de neurones (Deep Learning) sur les spectrogrammes des fichiers MP3 plutôt que de dépendre de l'API Spotify.
3. **Tester d'autres algorithmes :** Essayer des modèles avancés comme XGBoost ou LightGBM.
""")

st.divider()
st.success("🎉 Merci pour votre attention ! N'hésitez pas si vous avez des questions.")