import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from config import AUDIO_FEATURES
from data import clean_data, load_data

# Couleurs proches de la charte Spotify
FOND = "#121212"
PANNEAU = "#1C1C1C"
VERT = "#1DB954"
BLANC = "#F5F5F5"
GRIS = "#B3B3B3"
COULEURS_GENRES = ["#1DB954", "#55D88A", "#169C49", "#9CE5B4", "#318F60", "#78BFA0"]

st.set_page_config(page_title="Exploration des données", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #121212; color: #F5F5F5; }
    [data-testid="stSidebar"] { background-color: #181818; }
    [data-testid="stMetric"] {
        background: #1C1C1C;
        border: 1px solid #303030;
        border-radius: 10px;
        padding: 15px 18px;
    }
    h1, h2, h3 { letter-spacing: -0.02em; }
    h1 { color: #F5F5F5; }
    hr { border-color: #303030; }
    </style>
    """,
    unsafe_allow_html=True,
)


def style_graphique(fig, axes, grille=True):
    """Applique le même style sombre à tous les graphiques."""
    fig.patch.set_facecolor(FOND)
    for ax in np.array(axes, dtype=object).flat:
        ax.set_facecolor(PANNEAU)
        ax.tick_params(colors=GRIS, labelsize=9)
        ax.xaxis.label.set_color(BLANC)
        ax.yaxis.label.set_color(BLANC)
        ax.title.set_color(BLANC)
        for bordure in ax.spines.values():
            bordure.set_color("#353535")
        ax.set_axisbelow(True)
        if grille:
            ax.grid(axis="y", color="#505050", alpha=0.22)


def etiquettes_barres(ax, orientation="v", format_n="%.0f"):
    for groupe in ax.containers:
        if hasattr(groupe, "datavalues"):
            ax.bar_label(
                groupe,
                fmt=format_n,
                padding=3,
                color=BLANC,
                fontsize=9,
            )


# Chargement des données et filtre
st.title("Analyse exploratoire des données")
st.caption("Spotify · Exploration des caractéristiques musicales")

df_brut = load_data()
df = clean_data(df_brut)

st.sidebar.header("Filtres interactifs")
genres_dispos = sorted(df["playlist_genre"].dropna().unique().tolist())
genre_selectionne = st.sidebar.multiselect(
    "Sélectionnez un ou plusieurs genres",
    options=genres_dispos,
    default=genres_dispos,
)

df_filtre = df[df["playlist_genre"].isin(genre_selectionne)].copy()

if df_filtre.empty:
    st.info("Sélectionnez au moins un genre dans la barre latérale pour afficher les analyses.")
    st.stop()

st.subheader("Indicateurs clés du sous-échantillon")
col1, col2, col3 = st.columns(3)
col1.metric("Morceaux", f"{len(df_filtre):,}".replace(",", " "))
col2.metric("Durée moyenne (min)", f"{df_filtre['duration_min'].mean():.2f}")
col3.metric("Énergie moyenne", f"{df_filtre['energy'].mean():.2f}")
st.caption("Les indicateurs et graphiques sont actualisés en fonction des genres sélectionnés.")
st.divider()


# 1. Structure des données
st.subheader("1. Structure des données")
st.write(
    "Le jeu de données contient les informations des morceaux, leurs playlists "
    "et leurs caractéristiques audio. Les analyses suivantes utilisent les données nettoyées."
)
c1, c2, c3 = st.columns(3)
c1.metric("Lignes dans le fichier brut", f"{len(df_brut):,}".replace(",", " "))
c2.metric("Morceaux après nettoyage", f"{len(df):,}".replace(",", " "))
c3.metric("Colonnes après préparation", len(df.columns))
with st.expander("Voir un extrait des données et les valeurs manquantes"):
    st.dataframe(df_filtre.head(10), use_container_width=True, hide_index=True)
    manquantes = df_brut.isna().sum()
    st.write("Valeurs manquantes dans le fichier brut :")
    st.dataframe(
        manquantes[manquantes > 0].rename("Nombre de valeurs manquantes"),
        use_container_width=True,
    )


# 2. Doublons
st.subheader("2. Vérification des doublons")
nb_ids = df_brut["track_id"].nunique()
nb_multigenres = (
    df_brut.groupby("track_id")["playlist_genre"].nunique().gt(1).sum()
)
st.write(
    "Certains morceaux apparaissent plusieurs fois car ils sont présents dans différentes playlists. "
    "Le nettoyage retire les doublons d'identifiant ainsi que les entrées dont "
    "les métadonnées et caractéristiques audio sont identiques."
)
c1, c2, c3 = st.columns(3)
c1.metric("Identifiants uniques (brut)", f"{nb_ids:,}".replace(",", " "))
c2.metric("Lignes supprimées", f"{len(df_brut) - len(df):,}".replace(",", " "))
c3.metric("Titres liés à plusieurs genres", f"{nb_multigenres:,}".replace(",", " "))
st.caption(
    "Pour chaque groupe de doublons identifié, la version la plus populaire est conservée. "
    "Un même titre peut toutefois avoir plusieurs identifiants ou genres."
)


# 3. Transformations
st.subheader("3. Préparation des variables")
st.write(
    "Deux colonnes ont été ajoutées : l'année de sortie (`release_year`) et la durée "
    "en minutes (`duration_min`). Le genre musical (`playlist_genre`) est converti "
    "en catégorie. Ces traitements sont effectués dans `data.py`."
)
st.dataframe(
    df_filtre[["track_name", "release_year", "duration_min", "playlist_genre"]].head(8),
    use_container_width=True,
    hide_index=True,
)


# 4. Analyse univariée
st.subheader("4. Analyse univariée")
st.write(
    "On observe séparément la répartition des caractéristiques audio, "
    "de la popularité, de la durée et des genres musicaux."
)

st.markdown("**4.1 Caractéristiques audio**")
fig, axes = plt.subplots(4, 3, figsize=(14, 12))
for ax, variable in zip(axes.flat, AUDIO_FEATURES):
    ax.hist(df_filtre[variable].dropna(), bins=28, color=VERT, edgecolor=PANNEAU, alpha=0.95)
    ax.set_title(variable, fontsize=11, pad=9)
    ax.set_ylabel("Morceaux", fontsize=9)
for ax in list(axes.flat)[len(AUDIO_FEATURES):]:
    ax.set_visible(False)
style_graphique(fig, axes)
fig.tight_layout(pad=2)
st.pyplot(fig, use_container_width=True)
plt.close(fig)
st.caption("Ces histogrammes montrent les valeurs les plus fréquentes pour chaque variable audio.")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**4.2 Popularité**")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df_filtre["track_popularity"].dropna(), bins=30, color=VERT, edgecolor=PANNEAU)
    ax.set(xlabel="Popularité (0 à 100)", ylabel="Nombre de morceaux", xlim=(0, 100))
    style_graphique(fig, [ax])
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("Le score est celui enregistré dans ce dataset historique. La valeur 0 est conservée.")

with c2:
    st.markdown("**4.3 Durée des morceaux**")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    durees = df_filtre["duration_min"].dropna()
    durees_courtes = durees[durees <= 15]
    ax.hist(durees_courtes, bins=35, color=VERT, edgecolor=PANNEAU)
    ax.set(xlabel="Durée (minutes)", ylabel="Nombre de morceaux", xlim=(0, 15))
    style_graphique(fig, [ax])
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(f"Affichage limité à 15 min pour la lisibilité ({len(durees) - len(durees_courtes)} morceaux plus longs non affichés).")

st.markdown("**4.4 Répartition par genre**")
comptes_genres = df_filtre["playlist_genre"].value_counts().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.bar(comptes_genres.index.astype(str), comptes_genres.values, color=VERT, width=0.65)
ax.set(xlabel="Genre musical", ylabel="Nombre de morceaux")
ax.set_ylim(0, comptes_genres.max() * 1.13)
style_graphique(fig, [ax])
etiquettes_barres(ax)
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)
st.caption("Un morceau est associé à un seul genre après le nettoyage, même s'il figure dans plusieurs playlists.")

st.markdown("**4.5 Répartition par sous-genre**")
comptes_sous_genres = df_filtre["playlist_subgenre"].value_counts()
fig, ax = plt.subplots(figsize=(12, max(5, len(comptes_sous_genres) * 0.38)))
ax.barh(comptes_sous_genres.index.astype(str), comptes_sous_genres.values, color=VERT)
ax.invert_yaxis()
ax.set(xlabel="Nombre de morceaux", ylabel="Sous-genre")
ax.set_xlim(0, comptes_sous_genres.max() * 1.16)
style_graphique(fig, [ax], grille=False)
etiquettes_barres(ax, orientation="h")
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)
st.caption("Les sous-genres précisent les styles musicaux à l'intérieur des six genres principaux.")


# 5. Analyse bivariée
st.subheader("5. Analyse bivariée")
st.write("On compare maintenant plusieurs variables pour étudier leurs relations, notamment avec `energy`.")

st.markdown("**5.1 Corrélations entre caractéristiques audio**")
correlation = df_filtre[AUDIO_FEATURES].corr()
fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap=sns.diverging_palette(15, 145, s=70, l=45, as_cmap=True),
    vmin=-1,
    vmax=1,
    center=0,
    linewidths=0.25,
    linecolor="#313131",
    annot_kws={"size": 8},
    cbar_kws={"shrink": 0.8},
    ax=ax,
)
style_graphique(fig, [ax], grille=False)
ax.tick_params(axis="x", rotation=45)
ax.tick_params(axis="y", rotation=0)
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)
r_loudness = df_filtre["loudness"].corr(df_filtre["energy"])
r_acousticness = df_filtre["acousticness"].corr(df_filtre["energy"])
st.caption(
    f"Corrélation avec l'énergie : loudness = {r_loudness:+.2f} ; "
    f"acousticness = {r_acousticness:+.2f}. Une corrélation ne prouve pas une causalité."
)

st.markdown("**5.2 Caractéristiques audio selon le genre**")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, variable, etiquette in zip(axes, ["energy", "danceability"], ["Énergie", "Danceability"]):
    sns.boxplot(data=df_filtre, x="playlist_genre", y=variable, color=VERT, ax=ax)
    ax.set(xlabel="Genre musical", ylabel=etiquette, ylim=(0, 1))
    ax.set_title(etiquette + " selon le genre")
style_graphique(fig, axes)
fig.tight_layout(pad=2)
st.pyplot(fig, use_container_width=True)
plt.close(fig)
st.caption("Les boîtes comparent les médianes et la dispersion des valeurs selon les genres.")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**5.3 Popularité moyenne par genre**")
    moyennes = df_filtre.groupby("playlist_genre", observed=True)["track_popularity"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(moyennes.index.astype(str), moyennes.values, color=VERT)
    ax.set(xlabel="Genre musical", ylabel="Popularité moyenne", ylim=(0, 100))
    style_graphique(fig, [ax])
    etiquettes_barres(ax, format_n="%.1f")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("Moyenne des scores du dataset, y compris les valeurs égales à 0.")

with c2:
    st.markdown("**5.4 Popularité selon l'année de sortie**")
    popularite_annee = df_filtre.groupby("release_year").agg(
        popularite_moyenne=("track_popularity", "mean"),
        nombre_morceaux=("track_id", "size"),
    )
    popularite_annee = popularite_annee[popularite_annee["nombre_morceaux"] >= 30]
    if not popularite_annee.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(popularite_annee.index, popularite_annee["popularite_moyenne"], color=VERT, linewidth=2)
        ax.set(xlabel="Année de sortie", ylabel="Popularité moyenne", ylim=(0, 100))
        style_graphique(fig, [ax])
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Pas assez de morceaux par année avec ce filtre.")
    st.caption("Seules les années avec au moins 30 morceaux sont représentées.")

st.markdown("**5.5 Relation entre loudness et energy**")
echantillon = df_filtre.sample(n=min(3000, len(df_filtre)), random_state=42)
genres_echantillon = sorted(echantillon["playlist_genre"].dropna().unique().tolist())
palette = {genre: COULEURS_GENRES[i % len(COULEURS_GENRES)] for i, genre in enumerate(genres_echantillon)}
fig, ax = plt.subplots(figsize=(11, 6))
sns.scatterplot(
    data=echantillon,
    x="loudness",
    y="energy",
    hue="playlist_genre",
    hue_order=genres_echantillon,
    palette=palette,
    alpha=0.5,
    s=22,
    linewidth=0,
    ax=ax,
)
ax.set(xlabel="Loudness (dB)", ylabel="Énergie")
style_graphique(fig, [ax])
legende = ax.get_legend()
if legende is not None:
    legende.set_title("Genre musical")
    legende.get_frame().set_facecolor(PANNEAU)
    legende.get_frame().set_edgecolor("#414141")
    for texte in legende.get_texts():
        texte.set_color(BLANC)
    legende.get_title().set_color(BLANC)
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)
st.caption("Un échantillon de 3 000 morceaux maximum est affiché pour éviter la superposition des points.")


# 6. Artistes et titres
st.subheader("6. Artistes et morceaux les plus représentés")
st.write(
    "Ce classement présente les artistes les plus présents dans le fichier et les titres "
    "ayant les meilleurs scores de popularité. Il ne représente pas les écoutes actuelles sur Spotify."
)

st.markdown("**6.1 Top 10 des artistes les plus présents**")
top_artistes = df_filtre["track_artist"].value_counts().head(10)
fig, ax = plt.subplots(figsize=(11, 5))
ax.barh(top_artistes.index[::-1], top_artistes.values[::-1], color=VERT)
ax.set(xlabel="Nombre de morceaux", ylabel="Artiste")
ax.set_xlim(0, top_artistes.max() * 1.15)
style_graphique(fig, [ax], grille=False)
etiquettes_barres(ax, orientation="h")
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)
st.caption("Un artiste peut figurer plusieurs fois dans le jeu de données avec des titres différents.")

st.markdown("**6.2 Top 10 des morceaux les plus populaires**")
top_morceaux = (
    df_filtre.dropna(subset=["track_artist", "track_name"])
    .sort_values("track_popularity", ascending=False, kind="stable")
    .drop_duplicates(subset=["track_artist", "track_name"])
    .head(10)
    .copy()
)
top_morceaux["etiquette"] = (
    top_morceaux["track_name"].str.slice(0, 34)
    + " - "
    + top_morceaux["track_artist"].str.slice(0, 22)
)
fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(top_morceaux["etiquette"][::-1], top_morceaux["track_popularity"][::-1], color=VERT)
ax.set(xlabel="Popularité (0 à 100)", ylabel="Morceau", xlim=(0, 110))
style_graphique(fig, [ax], grille=False)
etiquettes_barres(ax, orientation="h")
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)
st.caption("Pour chaque titre et artiste, seule la version ayant le meilleur score est affichée dans le top 10.")


# 7. Récapitulatif
st.subheader("7. Récapitulatif")
nb_brut = f"{len(df_brut):,}".replace(",", " ")
nb_nettoye = f"{len(df):,}".replace(",", " ")
nb_multigenres_fr = f"{nb_multigenres:,}".replace(",", " ")
moyenne_dance = df_filtre["danceability"].mean()

st.markdown(
    f"""
    1. **Nettoyage :** {nb_brut} lignes au départ, {nb_nettoye} morceaux conservés.
    2. **Genres :** {nb_multigenres_fr} morceaux sont associés à plusieurs genres dans les données brutes.
    3. **Caractéristiques audio :** énergie moyenne de **{df_filtre['energy'].mean():.2f}** et danceability moyenne de **{moyenne_dance:.2f}** sur la sélection.
    4. **Loudness et énergie :** corrélation de **{r_loudness:+.2f}** sur la sélection.
    5. **Acousticness et énergie :** corrélation de **{r_acousticness:+.2f}** sur la sélection.
    """
)
