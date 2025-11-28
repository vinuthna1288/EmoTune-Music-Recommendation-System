import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# 1️⃣ Load your trained model (update the path to your .h5 model)
model_path = r"C:\Users\VINUTHNA\OneDrive\Desktop\INFOSYSPROJECT\INFOSYSPROJECT\ml\data\processed\emotion_cnn_model.h5"
model = load_model(model_path)

# 2️⃣ Define class labels (must match training order)
class_labels = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

# 3️⃣ Preprocess and predict function
def predict_emotion(img_path):
    # Load image in grayscale and resize
    img = image.load_img(img_path, target_size=(48, 48), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Predict emotion
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    return class_labels[predicted_class], confidence

# 4️⃣ Example usage
if __name__ == "__main__":
    # Update this with the image you want to test
    test_image = r"C:\Users\VINUTHNA\OneDrive\Desktop\infosysprojectwork\infosysprojectwork\ml\data\processed\test\happy\aug_554766.png"
    
    emotion, conf = predict_emotion(test_image)
    print(f"Predicted Emotion: {emotion} ({conf*100:.2f}% confidence)")
