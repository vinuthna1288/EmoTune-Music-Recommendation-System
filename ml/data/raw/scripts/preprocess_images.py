import cv2
import os
from tqdm import tqdm

raw_folder = r"C:\Users\VINUTHNA\OneDrive\Desktop\infosysprojectwork\infosysprojectwork\ml\data\raw\archive (2)"

processed_folder = r'C:\Users\VINUTHNA\OneDrive\Desktop\infosysprojectwork\infosysprojectwork\ml\data\processed'


os.makedirs(processed_folder, exist_ok=True)

for root, dirs, files in os.walk(raw_folder):
    for image_name in tqdm(files, desc=f"Processing in {root}"):
        if image_name.endswith(('.jpg', '.png')):
            image_path = os.path.join(root, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # already gray
            if img is None:
                continue
            resized_img = cv2.resize(img, (48, 48))  # ensure correct size
            normalized_img = resized_img / 255.0

            relative_path = os.path.relpath(root, raw_folder)
            save_dir = os.path.join(processed_folder, relative_path)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, image_name)

            cv2.imwrite(save_path, (normalized_img * 255).astype('uint8'))

print("Preprocessing complete!")
