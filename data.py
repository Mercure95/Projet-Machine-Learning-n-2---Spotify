import pandas as pd
import streamlit as st
from config import URL_SPOTIFY


@st.cache_data
def load_data():
    df = pd.read_csv(URL_SPOTIFY)
    return df

@st.cache_data
def clean_data(df):
    df_clean = df.copy()
    df_clean = df_clean.sort_values("track_popularity", ascending=False)

    # doublons en fonction du track id
    df_clean = df_clean.drop_duplicates(
        subset=["track_id"],
        keep="first"
    )

    # autres doublons
    df_clean = df_clean.drop_duplicates(
        subset=["track_artist", "track_name", "track_album_name", "duration_ms"],
        keep="first"
    )
    
    df_clean['release_year'] = pd.to_datetime(df_clean['track_album_release_date'], format='mixed').dt.year
    df_clean['duration_min'] = df_clean['duration_ms'] / 60000
    df_clean['playlist_genre'] = df_clean['playlist_genre'].astype('category')
    
    return df_clean