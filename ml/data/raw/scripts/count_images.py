import os

# Path to your processed data
processed_path = r"C:\Users\VINUTHNA\OneDrive\Desktop\infosysprojectwork\infosysprojectwork\ml\data\processed"

splits = ["train", "val", "test"]

print("| Class | Train | Val | Test |")
print("|-------|-------|-----|------|")

# Get all emotion classes from the train folder
classes = [d for d in os.listdir(os.path.join(processed_path, "train")) if os.path.isdir(os.path.join(processed_path, "train", d))]

for cls in classes:
    counts = []
    for split in splits:
        split_path = os.path.join(processed_path, split, cls)
        if os.path.exists(split_path):
            counts.append(len([f for f in os.listdir(split_path) if f.endswith((".jpg", ".png"))]))
        else:
            counts.append(0)
    print(f"| {cls} | {counts[0]} | {counts[1]} | {counts[2]} |")
