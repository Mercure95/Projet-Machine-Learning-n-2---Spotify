import pandas as pd
import streamlit as st
from config import AUDIO_FEATURES, URL_SPOTIFY


@st.cache_data
def load_data():
    df = pd.read_csv(URL_SPOTIFY)
    return df

@st.cache_data
def clean_data(df):
    df_clean = df.copy()

    df_clean = df_clean.sort_values(
        "track_popularity",
        ascending=False,
        kind="stable"
    )

    # doublons de track_id
    df_clean = df_clean.drop_duplicates(
        subset=["track_id"],
        keep="first"
    )

    # doublons autres
    df_clean = df_clean.drop_duplicates(
        subset=[
            "track_artist",
            "track_name",
            "track_album_name",
            "duration_ms"
        ] + AUDIO_FEATURES,
        keep="first"
    )

    # Partie 1 : Section 3
    df_clean['release_year'] = pd.to_datetime(df_clean['track_album_release_date'], format='mixed').dt.year
    df_clean['duration_min'] = df_clean['duration_ms'] / 60000
    df_clean['playlist_genre'] = df_clean['playlist_genre'].astype('category')
    
    return df_clean