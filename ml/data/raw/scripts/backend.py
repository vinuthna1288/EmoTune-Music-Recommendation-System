# --- EmoTune Backend (Flask + Fallback Multilingual) ---
import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import cv2

app = Flask(__name__)

# --- Load trained model ---
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best_model.h5"))
model = load_model(MODEL_PATH)

# --- Emotion Labels ---
EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

# --- Load Haar Cascade for face detection ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --- Fallback Tracks (all emotions + languages) ---
FALLBACK_TRACKS = {
    "anger": {
        "english": [{"name": "Believer", "artist": "Imagine Dragons", "album": "Evolve",
                     "spotify": "https://open.spotify.com/track/0pqnGHJpmpxLKifKRmU6WP"}],
        "telugu": [{"name": "Naatu Naatu", "artist": "Rahul Sipligunj, Kaala Bhairava", "album": "RRR",
                    "spotify": "https://open.spotify.com/track/4iKGu3xtvm90eBw0EIPWJP?si=8d1b6612e8ef44ed"}],
        "hindi": [{"name": "Ghungroo", "artist": "Arijit Singh, Shilpa Rao", "album": "War",
                   "spotify": "https://open.spotify.com/track/6EAMI8iHwC3TCSBGVKoWng?si=ced64c5384ae44b7"}],
        "tamil": [{"name": "Verithanam", "artist": "A.R. Rahman", "album": "Bigil",
                   "spotify": "https://open.spotify.com/track/6YhDs8isyaiVECBSViAFDS?si=48ece2272a544db0"}]
    },
    "happiness": {
        "english": [{"name": "Happy", "artist": "Pharrell Williams", "album": "G I R L",
                     "spotify": "https://open.spotify.com/track/60nZcImufyMA1MKQY3dcCH"}],
        "telugu": [{"name": "Butta Bomma", "artist": "Armaan Malik", "album": "Ala Vaikunthapurramuloo",
                    "spotify": "https://open.spotify.com/track/0dnDTvdUco2UbaBjUtPxNS?si=bf6dfba5ced64edd"}],
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
        "english": [{"name": "ocean eyes", "artist": "Owl City", "album": "Billie Eilish",
                     "spotify": "https://open.spotify.com/track/7hDVYcQq6MxkdJGweuCtl9?si=bde61ec8614d436a"}],
        "telugu": [{"name": "Karige Loga", "artist": "Devi Sri Prasad", "album": "Aarya 2",
                    "spotify": "https://open.spotify.com/track/37I2SLeJcpkQpKFMLJt550?si=699c5f56c87d4bc7"}],
        "hindi": [{"name": "Channa Mereya", "artist": "Arijit Singh", "album": "Ae Dil Hai Mushkil",
                   "spotify": "https://open.spotify.com/track/0H2iJVgorRR0ZFgRqGUjUM?si=6c38280ac52943b0"}]
    },
    "surprise": {
        "english": [{"name": "Levels", "artist": "Avicii", "album": "Levels",
                     "spotify": "https://open.spotify.com/track/6jJ0s89eD6GaHleKKya26X"}],
        "telugu": [{"name": "Dosti", "artist": "Anirudh Ravichander", "album": "RRR",
                    "spotify": "https://open.spotify.com/track/6h5vL07MyGY6WdWPYK4IMG?si=e3f3ab52104c48ac"}],
        "hindi": [{"name": "Malang Title Track", "artist": "Ved Sharma", "album": "Malang",
                   "spotify": "https://open.spotify.com/track/25MPTnqXQB1H6OkwSYUXWx?si=aebafeb1fd9340c0"}],
        "tamil": [{"name": "Verithanam", "artist": "A.R. Rahman", "album": "Bigil",
                   "spotify": "https://open.spotify.com/track/6YhDs8isyaiVECBSViAFDS?si=e0bbb4ff6f8d45d4"}]
    },
}

# --- Helper: preprocess face image ---
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        raise ValueError("No face detected")
    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = np.expand_dims(face, axis=(0, -1))
    return face

# --- Route: detect emotion ---
@app.route("/detect-emotion", methods=["POST"])
def detect_emotion():
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image uploaded"}), 400

        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = np.array(image)
        face = preprocess_image(image)

        preds = model.predict(face)
        emotion_idx = int(np.argmax(preds))
        emotion = EMOTIONS[emotion_idx]
        confidence = float(np.max(preds))

        return jsonify({"emotion": emotion, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Route: get recommendations ---
@app.route("/recommendations", methods=["POST"])
def recommendations():
    try:
        data = request.get_json()
        emotion = data.get("emotion", "neutral").lower()
        language = data.get("language", "english").lower()  # lowercase to match fallback keys

        # Spotify first
        tracks = get_spotify_tracks(emotion)

        # Fallback tracks
        if not tracks or len(tracks) == 0:
            tracks_by_emotion = FALLBACK_TRACKS.get(emotion, FALLBACK_TRACKS.get("neutral", {}))
            tracks = tracks_by_emotion.get(language, tracks_by_emotion.get("english", []))

        if not tracks or len(tracks) == 0:
            return jsonify({"error": "No songs found for this emotion and language."}), 404

        return jsonify({"emotion": emotion, "language": language, "tracks": tracks})

    except Exception as e:
        return jsonify({"error": str(e)}), 500





# ---------- EMOTION DETECTION ENDPOINT ----------
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Load your trained model once globally
model = load_model("emotion_model.h5")
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

@app.route("/detect-emotion", methods=["POST"])
def detect_emotion():
    try:
        file = request.files['image']             # read the uploaded image
        img = Image.open(file.stream).convert("L")  # convert to grayscale
        img = img.resize((48, 48))                  # resize for CNN
        arr = np.expand_dims(np.expand_dims(np.array(img) / 255.0, -1), 0)

        preds = model.predict(arr)
        idx = int(np.argmax(preds))
        emotion = EMOTIONS[idx]
        confidence = float(np.max(preds))

        return jsonify({"emotion": emotion, "confidence": confidence})
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500
