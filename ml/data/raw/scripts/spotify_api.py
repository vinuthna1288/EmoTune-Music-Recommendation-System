import os
import requests
from dotenv import load_dotenv

# --- Load keys from .env ---
load_dotenv()
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# --- Get Spotify access token ---
def get_spotify_token():
    auth_url = "https://accounts.spotify.com/api/token"
    response = requests.post(auth_url, {
        'grant_type': 'client_credentials',
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET
    })
    response.raise_for_status()
    return response.json()["access_token"]

# --- Map emotions to genres ---
EMOTION_GENRE_MAP = {
    "Happy": "pop",
    "Sad": "acoustic",
    "Angry": "rock",
    "Surprise": "electronic",
    "Fear": "ambient",
    "Disgust": "metal",
    "Neutral": "chill"
}

# --- Fetch tracks for an emotion ---
def get_tracks_for_emotion(emotion):
    token = get_spotify_token()
    genre = EMOTION_GENRE_MAP.get(emotion, "pop")
    url = f"https://api.spotify.com/v1/search?q=genre:{genre}&type=track&limit=5"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    tracks = []
    for item in data["tracks"]["items"]:
        tracks.append({
            "title": item["name"],
            "artist": item["artists"][0]["name"],
            "album_art": item["album"]["images"][0]["url"] if item["album"]["images"] else None,
            "preview_url": item["preview_url"]
        })
    return tracks
