import os

# Path to processed data
processed_folder = r"C:\Users\VINUTHNA\OneDrive\Desktop\infosysprojectwork\infosysprojectwork\ml\data\processed"

splits = ["train", "val", "test"]

print("📊 Dataset Report\n")

for split in splits:
    split_path = os.path.join(processed_folder, split)
    print(f"--- {split.upper()} ---")
    
    if not os.path.exists(split_path):
        print(f"❌ {split} folder not found!\n")
        continue

    total_images = 0
    for emotion in os.listdir(split_path):
        emotion_path = os.path.join(split_path, emotion)
        if not os.path.isdir(emotion_path):
            continue

        count = len(os.listdir(emotion_path))
        total_images += count
        print(f"{emotion}: {count} images")

    print(f"TOTAL in {split}: {total_images} images\n")
