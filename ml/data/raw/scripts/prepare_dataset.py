import os
import shutil
import random
from tqdm import tqdm

# Paths
processed_folder = r"C:\Users\VINUTHNA\OneDrive\Desktop\INFOSYSPROJECT\INFOSYSPROJECT\ml\data\processed"
output_folder = r"C:\Users\VINUTHNA\OneDrive\Desktop\INFOSYSPROJECT\INFOSYSPROJECT\ml\data\dataset"

# Split ratios
train_split = 0.7
val_split = 0.15
test_split = 0.15

# Create output folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_folder, split), exist_ok=True)

# Get emotion folders
emotions = [d for d in os.listdir(processed_folder) if os.path.isdir(os.path.join(processed_folder, d))]

print(f"Found emotion classes: {emotions}")

# For each emotion folder
for emotion in emotions:
    emotion_path = os.path.join(processed_folder, emotion)
    images = [f for f in os.listdir(emotion_path) if f.endswith(('.jpg', '.png'))]
    random.shuffle(images)

    total = len(images)
    train_end = int(train_split * total)
    val_end = int((train_split + val_split) * total)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, split_files in splits.items():
        split_dir = os.path.join(output_folder, split, emotion)
        os.makedirs(split_dir, exist_ok=True)
        for img_name in tqdm(split_files, desc=f"{emotion} → {split}"):
            src = os.path.join(emotion_path, img_name)
            dst = os.path.join(split_dir, img_name)
            shutil.copy(src, dst)

print("\n✅ Dataset preparation complete!")
print(f"Data saved in: {output_folder}")
