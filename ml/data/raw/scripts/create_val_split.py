import os
import random
import shutil

# Paths
processed_folder = r"C:\Users\VINUTHNA\OneDrive\Desktop\infosysprojectwork\infosysprojectwork\ml\data\processed"
train_folder = os.path.join(processed_folder, "train")
val_folder = os.path.join(processed_folder, "val")

# Create val folder if not exists
os.makedirs(val_folder, exist_ok=True)

# Ratio for validation
val_ratio = 0.15

# For each emotion class
for emotion in os.listdir(train_folder):
    emotion_train_path = os.path.join(train_folder, emotion)
    if not os.path.isdir(emotion_train_path):
        continue

    emotion_val_path = os.path.join(val_folder, emotion)
    os.makedirs(emotion_val_path, exist_ok=True)

    images = os.listdir(emotion_train_path)
    random.shuffle(images)

    val_count = int(len(images) * val_ratio)
    val_images = images[:val_count]

    for img in val_images:
        src = os.path.join(emotion_train_path, img)
        dst = os.path.join(emotion_val_path, img)
        shutil.move(src, dst)  # move instead of copy

print("✅ Validation set created successfully!")
