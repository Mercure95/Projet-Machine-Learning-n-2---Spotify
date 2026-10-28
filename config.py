URL_SPOTIFY = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET_REG = "energy"
TARGET_CLF = "playlist_genre"

AUDIO_FEATURES = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "loudness",
    "tempo", "key", "mode"
]

FEATURES_REG = [
    "danceability", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "loudness",
    "tempo", "key", "mode", "duration_min", "release_year"
]

FEATURES_CLF = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "loudness",
    "tempo", "key", "mode", "duration_min", "release_year"
]