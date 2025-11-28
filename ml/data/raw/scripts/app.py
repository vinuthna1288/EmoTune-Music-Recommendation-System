import streamlit as st
import os
import json
import requests
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import hashlib


# Add custom font + CSS theme tweaks
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">',
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background: linear-gradient(135deg, #0f1724, #121428 60%, #0e1a2b);
        color: #FAFAFA;
        font-family: 'Poppins', sans-serif;
    }

    /* Center main content width */
    .main .block-container{
        max-width: 900px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* Headings */
    h1, h2, h3, h4 {
        text-align: center;
        color: #FF4081 !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: #FF4081;
        color: white;
        border-radius: 10px;
        height: 3rem;
        width: 100%;
        border: none;
        font-weight: 600;
    }
    .stButton>button:hover { transform: translateY(-2px); }

    /* Inputs */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #1f2636;
        color: #fff;
        border-radius: 8px;
    }

    /* Card for songs */
    .song-card {
        background-color: #131622;
        padding: 14px;
        border-radius: 12px;
        margin-bottom: 12px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.4);
    }

    /* Small meta text */
    .meta { color: #9aa3b2; font-size: 13px; }

    /* Info box style */
    .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)




# ------------------ Config & Helpers for Credentials ------------------

# Determine base dir reliably
try:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    BASE_DIR = os.getcwd()

CREDENTIALS_PATH = os.path.join(BASE_DIR, "credentials.json")

def hash_password(password: str) -> str:
    """Return SHA256 hex digest of the password."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def load_credentials() -> dict:
    """Load credentials from credentials.json or create default if missing."""
    if not os.path.exists(CREDENTIALS_PATH):
        # Create default credentials
        default = {"user": hash_password("123")}
        try:
            with open(CREDENTIALS_PATH, "w", encoding="utf-8") as f:
                json.dump(default, f, indent=2)
        except Exception as e:
            st.warning(f"Warning: couldn't create credentials file: {e}")
        return default
    try:
        with open(CREDENTIALS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        st.error(f"Failed to read credentials file: {e}")
        return {}

def save_credentials(creds: dict):
    """Save credentials dict to file."""
    try:
        with open(CREDENTIALS_PATH, "w", encoding="utf-8") as f:
            json.dump(creds, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save credentials: {e}")

def check_credentials(username: str, password: str) -> bool:
    creds = load_credentials()
    hashed = hash_password(password)
    stored = creds.get(username)
    return stored == hashed

def reset_password_for_user(username: str, new_password: str):
    creds = load_credentials()
    creds[username] = hash_password(new_password)
    save_credentials(creds)

# ------------------ Streamlit App Start ------------------

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "forgot_mode" not in st.session_state:
    st.session_state["forgot_mode"] = False

# Page config
st.set_page_config(page_title="EmoTune Login", page_icon="🎧", layout="centered")

# ------------------ LOGIN / FORGOT UI ------------------

# ------------------ Custom Label Styles ------------------
st.markdown("""
<style>
/* Style the field labels (like Username, Password, etc.) */
label[data-testid="stWidgetLabel"] > div > p {
    color: #FF4081 !important;       /* Pink — change this to your desired color */
    font-weight: 700;                /* Make the labels bold */
    font-size: 1.05rem;              /* Slightly larger text */
}

/* Optional: style the small "Forgot Password" and form headings too */
h3, h4, .stMarkdown p {
    color: #333333;
}
</style>
""", unsafe_allow_html=True)


# ------------------ Custom Button Styles ------------------
st.markdown("""
<style>
/* Common button base */
.stButton>button {
    font-weight: 600;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    border: none;
    cursor: pointer;
    transition: 0.2s all ease-in-out;
}

/* Login button (first one) */
div[data-testid="stFormSubmitButton"]:nth-child(1) button {
    background-color: #4CAF50; /* Green */
    color: white;
}
div[data-testid="stFormSubmitButton"]:nth-child(1) button:hover {
    background-color: #45a049;
    transform: scale(1.05);
}

/* Forgot Password button (second one) */
div[data-testid="stFormSubmitButton"]:nth-child(2) button {
    background-color: #f44336; /* Red */
    color: white;
}
div[data-testid="stFormSubmitButton"]:nth-child(2) button:hover {
    background-color: #d32f2f;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)


if not st.session_state["logged_in"]:
    st.markdown(
        """
        <h1 style='text-align:center; color:#4CAF50;'>🎵 Welcome to <span style="color:#FF4081;">EmoTune</span></h1>
        <p style='text-align:center; color:gray;'>Emotion-based music recommendation system</p>
        """,
        unsafe_allow_html=True
    )

    # Forgot password screen: ask username + new pwd + confirm
    if st.session_state["forgot_mode"]:
        st.markdown("### 🔑 Forgot / Reset Password (no email required)")
        with st.form("forgot_form", clear_on_submit=False):
            fp_username = st.text_input("👤 Username")
            new_password = st.text_input("🔒 New Password", type="password")
            confirm_password = st.text_input("✅ Confirm Password", type="password")
            submitted = st.form_submit_button("Reset Password 🔁")

        if submitted:
            if not fp_username or not new_password or not confirm_password:
                st.warning("⚠️ Please fill all fields.")
            elif new_password != confirm_password:
                st.error("❌ Passwords do not match.")
            else:
                # Persist the password for that username
                reset_password_for_user(fp_username, new_password)
                st.success(f"✅ Password reset successful for user **{fp_username}**.")
                st.info("You can now log in with the new password.")
                st.session_state["forgot_mode"] = False
                st.experimental_rerun()

        if st.button("⬅️ Back to Login"):
            st.session_state["forgot_mode"] = False
            st.experimental_rerun()

        st.stop()

    # Login screen
    st.markdown("### 🔒 Please log in to continue")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("👤 Username")
        password = st.text_input("🔑 Password", type="password")
        col1, col2 = st.columns([3, 1])
        login_pressed = col1.form_submit_button("Login 🚀")
        forgot_pressed = col2.form_submit_button("Forgot Password?")

    if forgot_pressed:
        st.session_state["forgot_mode"] = True
        st.experimental_rerun()

    if login_pressed:
        if not username or not password:
            st.error("Please enter both username and password.")
        else:
            if check_credentials(username, password):
                st.session_state["logged_in"] = True
                st.success("✅ Login successful!")
                st.balloons()
                st.rerun()

            else:
                st.error("❌ Invalid username or password.")
    st.stop()

# ------------------ REST OF APP (after login) ------------------

st.title("🎵 EmoTune — Emotion-Based Music Recommender")

# Backend URL
BACKEND_URL = "http://127.0.0.1:5000"

# Load Emotion Model (adjust your MODEL_PATH)
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best_model.h5"))

@st.cache_resource
def load_emotion_model():
    return tf.keras.models.load_model(MODEL_PATH)

# load model safely (catch exceptions to avoid crashing UI)
try:
    model = load_emotion_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.warning(f"Model load warning: {e}")
    model = None

EMOTIONS = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

def preprocess_image(pil_image):
    img = pil_image.convert("L")  # grayscale
    img = ImageOps.fit(img, (48, 48), Image.Resampling.LANCZOS)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def predict_emotion(pil_image):
    if model is None:
        raise RuntimeError("Emotion model not loaded.")
    processed = preprocess_image(pil_image)
    preds = model.predict(processed)
    emotion_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    return EMOTIONS[emotion_idx], confidence

# --- Fallback tracks (full dictionary) ---
FALLBACK_TRACKS = {
    "anger": {
        "english": [{"name": "Believer", "artist": "Imagine Dragons", "album": "Evolve",
                     "spotify": "https://open.spotify.com/track/0pqnGHJpmpxLKifKRmU6WP"}],
        "telugu": [{"name": "Hungry Cheetah", "artist": "Raghu Ram,Thaman S", "album": "They Call Him OG",
                    "spotify": "https://open.spotify.com/track/538V6gFAnhcLZrg6I8lIMI?si=ea84210300644a0f"}],
        "hindi": [{"name": "Ghungroo", "artist": "Arijit Singh, Shilpa Rao", "album": "War",
                   "spotify": "https://open.spotify.com/track/6EAMI8iHwC3TCSBGVKoWng?si=ced64c5384ae44b7"}],
        "tamil": [{"name": "Verithanam", "artist": "A.R. Rahman", "album": "Bigil",
                   "spotify": "https://open.spotify.com/track/6YhDs8isyaiVECBSViAFDS?si=48ece2272a544db0"}]
    },
    "happiness": {
        "english": [{"name": "Happy", "artist": "Pharrell Williams", "album": "G I R L",
                     "spotify": "https://open.spotify.com/track/60nZcImufyMA1MKQY3dcCH"}],
        "telugu": [{"name": "Ola Olaala Ala", "artist": "Karunya,Ranina Reddy", "album": "Orange",
                    "spotify": "https://open.spotify.com/track/2Z6PDgrkYCjASnRqWZxf4Y?si=f129ac0125ae46e9"}],
        "hindi": [{"name": "Gallan Goodiyan", "artist": "Yashita, Farhan, Shankar, Sukhwinder", "album": "Dil Dhadakne Do",
                   "spotify": "https://open.spotify.com/track/7hNYvX0qAKrxtVr1jGDmvR?si=585f7619f8bb453b"}],
        "tamil": [{"name": "Vaathi Coming", "artist": "Anirudh Ravichander", "album": "Master",
                   "spotify": "https://open.spotify.com/track/4dJrjWtAhEkW7VdPYSL1Ip?si=bc2ff701d5a346fd"}]
    },
    "disgust": {
        "english": [{"name": "Toxic", "artist": "Britney Spears", "album": "In the Zone",
                     "spotify": "https://open.spotify.com/track/6I9VzXrHxO9rA9A5euc8Ak"}],
        "telugu": [{"name": "Top Lesi Poddi", "artist": "DSP", "album": "Iddarammayilatho",
                    "spotify": "https://open.spotify.com/track/0FSrCEox8bEpO3RD6ZfEr5?si=329d28258b7646f0"}],
        "hindi": [{"name": "Kaala Chashma", "artist": "Badshah", "album": "Baar Baar Dekho",
                   "spotify": "https://open.spotify.com/track/6mdLX10dvBb7rGYbMXpKzz?si=377c9e24233449a4"}],
        "tamil": [{"name": "Kutti Story", "artist": "Anirudh Ravichander", "album": "Master",
                   "spotify": "https://open.spotify.com/track/0Xm3PXjA4kXu2GxBhkLk61?si=0ab612ce0ea6450c"}]
    },
    "fear": {
        "english": [{"name": "Thriller", "artist": "Michael Jackson", "album": "Thriller",
                     "spotify": "https://open.spotify.com/track/3S2R0EVwBSAVMd5UMgKTL0?si=859624e299f24522"}],
        "telugu": [{"name": "Fear Song", "artist": "Anirudh Ravichander", "album": "Devara",
                    "spotify": "https://open.spotify.com/track/6b2WJDzGt5X8dYfpkWtvXW?si=676136bb3a084513"}],
        "hindi": [{"name": "Aazaadiyan", "artist": "Mohit Chauhan", "album": "Udaan",
                   "spotify": "https://open.spotify.com/track/4vLEY203BopDMTh3IwzDEn?si=ed429f033bcc4732"}],
        "tamil": [{"name": "Anirudh Scary", "artist": "Anirudh Ravichander", "album": "Kaithi",
                   "spotify": "https://open.spotify.com/track/0JGTW6P8nL1ZFCSp7H0nwC?si=aecffebb8ef8483c"}]
    },
    "neutral": {
        "english": [{"name": "Counting Stars", "artist": "OneRepublic", "album": "Native",
                     "spotify": "https://open.spotify.com/track/2tpWsVSb9UEmDRxAl1zhX1"}],
        "telugu": [{"name": "Manasa Manasa", "artist": "Sid Sriram", "album": "Most Eligible Bachelor",
                    "spotify": "https://open.spotify.com/track/28GsFcWboBzq0kVMXl0cDL?si=eb68026815274947"}],
        "hindi": [{"name": "Raabta", "artist": "Arijit Singh", "album": "Raabta",
                   "spotify": "https://open.spotify.com/track/6FjbAnaPRPwiP3sciEYctO?si=fa30f38bf66f4607"}],
        "tamil": [{"name": "Anbil Avan", "artist": "Sid Sriram", "album": "Pariyerum Perumal",
                   "spotify": "https://open.spotify.com/track/1QuZBM0iHDlr1oRVyeZypC?si=1ae61a21a8d74fbf"}]
    },
    "sadness": {
        "english": [{"name": "Ocean Eyes", "artist": "Owl City", "album": "Billie Eilish",
                     "spotify": "https://open.spotify.com/track/7hDVYcQq6MxkdJGweuCtl9?si=bde61ec8614d436a"}],
        "telugu": [{"name": "Karige Loga", "artist": "Devi Sri Prasad", "album": "Aarya 2",
                    "spotify": "https://open.spotify.com/track/37I2SLeJcpkQpKFMLJt550?si=699c5f56c87d4bc7"}],
        "hindi": [{"name": "Channa Mereya", "artist": "Arijit Singh", "album": "Ae Dil Hai Mushkil",
                   "spotify": "https://open.spotify.com/track/0H2iJVgorRR0ZFgRqGUjUM?si=6c38280ac52943b0"}],
        "tamil": [{"name": "Anbil Avan Sad", "artist": "Sid Sriram", "album": "Sad Tamil Song",
                   "spotify": "https://open.spotify.com/track/1QuZBM0iHDlr1oRVyeZypC?si=bcbfb43128e34d4d"}]
    },
    "surprise": {
        "english": [{"name": "Levels", "artist": "Avicii", "album": "Levels",
                     "spotify": "https://open.spotify.com/track/5UqCQaDshqbIk3pkhy4Pjg?si=1c2e174d0d394666"}],
        "telugu": [{"name": "Dosti", "artist": "Anirudh Ravichander", "album": "RRR",
                    "spotify": "https://open.spotify.com/track/6h5vL07MyGY6WdWPYK4IMG?si=e3f3ab52104c48ac"}],
        "hindi": [{"name": "Malang Title Track", "artist": "Ved Sharma", "album": "Malang",
                   "spotify": "https://open.spotify.com/track/25MPTnqXQB1H6OkwSYUXWx?si=aebafeb1fd9340c0"}],
        "tamil": [{"name": "Verithanam", "artist": "A.R. Rahman", "album": "Bigil",
                   "spotify": "https://open.spotify.com/track/6YhDs8isyaiVECBSViAFDS?si=e0bbb4ff6f8d45d4"}]
    },
}

# ------------------ Functions to fetch and fallback ------------------

def get_song_recommendations(emotion, language):
    try:
        payload = {"emotion": emotion.lower(), "language": language.lower()}
        response = requests.post(f"{BACKEND_URL}/recommendations", json=payload, timeout=5)
        if response.status_code == 200:
            return response.json().get("tracks", [])
    except Exception:
        pass
    return []

def get_song_recommendations_with_fallback(emotion, language):
    tracks = get_song_recommendations(emotion, language)
    if not tracks:
        emotion_key = emotion.lower()
        language_key = language.lower()
        tracks_by_emotion = FALLBACK_TRACKS.get(emotion_key, FALLBACK_TRACKS.get("neutral", {}))
        tracks = tracks_by_emotion.get(language_key, tracks_by_emotion.get("english", []))
    return tracks

# ------------------ UI for uploading / camera and displaying songs ------------------

st.markdown("## 😄 Detect Emotion and Get Music Recommendations")
st.markdown("Upload or capture your face image to detect emotion and instantly play songs 🎧")
st.markdown("---")

# --- Image Input Section ---
st.markdown("### 🖼️ Upload or Capture an Image")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

with col2:
    # Initialize camera state if not already done
    if "camera_on" not in st.session_state:
        st.session_state.camera_on = False

    if not st.session_state.camera_on:
        if st.button("📸 Turn On Camera"):
            st.session_state.camera_on = True
            st.rerun()
    else:
        if st.button("❌ Turn Off Camera"):
            st.session_state.camera_on = False
            st.rerun()

camera_image = None
if st.session_state.camera_on:
    st.info("📷 Camera is ON — capture your image below 👇")
    camera_image = st.camera_input("")

# --- Language Selection Section ---
st.markdown("### 🌐 Choose Your Music Language")
language = st.selectbox("", ["English", "Hindi", "Telugu", "Tamil"])

# --- Input selection priority ---
input_image = uploaded_file or camera_image

if input_image:
    try:
        image = Image.open(input_image)
        st.image(image, caption="Your Image", use_container_width=True)

        # Add spinner while emotion is detected
        with st.spinner("🔍 Detecting emotion... Please wait..."):
            emotion, confidence = predict_emotion(image)

        st.success(f"Detected Emotion: **{emotion.capitalize()}** (Confidence: {confidence:.2f})")

        # --- Keep your original song recommendation block (unchanged) ---
        tracks = get_song_recommendations_with_fallback(emotion, language)
        if tracks:
            st.subheader("🎧 Recommended Songs")
            for t in tracks:
                name = t.get("name", "Unknown")
                artist = t.get("artist", "Unknown")
                spotify_link = t.get("spotify") or t.get("url")

                st.write(f"**{name}** — {artist}")
                if spotify_link and "open.spotify.com/track" in spotify_link:
                    track_id = spotify_link.split("/")[-1].split("?")[0]
                    embed_url = f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator"
                    st.components.v1.iframe(embed_url, height=80)
                else:
                    st.info("⚠️ No valid Spotify link available for this track.")
        else:
            st.warning("No songs found for this emotion and language.")
    except Exception as e:
        st.error(f"Error detecting emotion: {str(e)}")
