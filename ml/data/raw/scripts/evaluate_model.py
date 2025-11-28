import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json

# ---------------- PATHS ----------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "models")  # Your model folder
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
LOGS_DIR = os.path.join(PROJECT_ROOT, "data", "logs")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(os.path.join(LOGS_DIR, "figures"), exist_ok=True)

# ---------------- MODEL ----------------
model_path = os.path.join(MODEL_DIR, "final_model.h5")  # or best_model.h5
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at: {model_path}")

print(f"🔍 Loading model from {model_path}")
model = tf.keras.models.load_model(model_path)

# ---------------- TEST DATA ----------------
TEST_DIR = os.path.join(DATA_DIR, "test")
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(48, 48),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=32,
    shuffle=False
)

# ---------------- PREDICTIONS ----------------
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

# ---------------- CLASSIFICATION REPORT ----------------
class_names = list(test_generator.class_indices.keys())
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# Save metrics as JSON
metrics_path = os.path.join(LOGS_DIR, "evaluation_results.json")
with open(metrics_path, "w") as f:
    json.dump(report_dict, f, indent=4)
print(f"\n✅ Metrics saved at {metrics_path}")

# ---------------- PER-CLASS ACCURACY ----------------
print("\n🎯 Per-class Accuracy:")
for label, metrics in report_dict.items():
    if label not in ['accuracy', 'macro avg', 'weighted avg']:
        print(f"{label}: {metrics['recall']*100:.2f}%")
overall_accuracy = report_dict['accuracy']*100
print(f"\n⭐ Overall Test Accuracy: {overall_accuracy:.2f}%")

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
cm_path = os.path.join(LOGS_DIR, "figures", "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print(f"✅ Confusion matrix saved at: {cm_path}")

# ---------------- SAMPLE PREDICTIONS ----------------
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flatten()):
    if i >= len(test_generator.filenames):
        break
    img, label = test_generator[i]
    ax.imshow(img[0].reshape(48, 48), cmap="gray")
    true_label = class_names[np.argmax(label[0])]
    pred_label = class_names[y_pred[i]]
    ax.set_title(f"T: {true_label}\nP: {pred_label}", fontsize=8)
    ax.axis("off")
sample_preds_path = os.path.join(LOGS_DIR, "figures", "sample_predictions.png")
plt.tight_layout()
plt.savefig(sample_preds_path)
plt.close()
print(f"✅ Sample predictions saved at: {sample_preds_path}")
