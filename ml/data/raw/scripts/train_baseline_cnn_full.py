import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score

# ---------------- PATHS ----------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "models")  # Matches your current folder
LOGS_DIR = os.path.join(PROJECT_ROOT, "data", "logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.h5")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
CSV_LOG_PATH = os.path.join(LOGS_DIR, "training_log.csv")

# ---------------- DATA AUGMENTATION ----------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(rescale=1./255)

# ---------------- GENERATORS ----------------
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(48, 48),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=64,
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(48, 48),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=64,
    shuffle=False
)

# ---------------- CLASS WEIGHTS ----------------
classes = np.unique(train_generator.classes)
class_weights_values = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=train_generator.classes
)
class_weights = dict(zip(classes, class_weights_values))
print("Class weights:", class_weights)

# ---------------- MODEL ----------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# ---------------- COMPILE ----------------
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------- CALLBACKS ----------------
checkpoint = ModelCheckpoint(BEST_MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=20, verbose=1, restore_best_weights=True)
csv_logger = CSVLogger(CSV_LOG_PATH, append=True)

# ---------------- TRAIN ----------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=120,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop, csv_logger]
)

# ---------------- SAVE FINAL MODEL ----------------
model.save(FINAL_MODEL_PATH)
print(f"✅ Training finished. Final model saved at: {FINAL_MODEL_PATH}")
print(f"✅ Best model saved at: {BEST_MODEL_PATH}")

# ---------------- EVALUATE PER-CLASS ACCURACY ----------------
print("\n📊 Evaluating per-emotion accuracy on validation set...")

val_generator.reset()
y_pred_probs = model.predict(val_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_generator.classes
labels = list(val_generator.class_indices.keys())

# per-class accuracy
for i, label in enumerate(labels):
    idx = np.where(y_true == i)[0]
    acc = np.mean(y_pred[idx] == y_true[idx])
    print(f"{label}: {acc*100:.2f}%")

# overall accuracy
overall_acc = accuracy_score(y_true, y_pred)
print(f"\n⭐ Overall Validation Accuracy: {overall_acc*100:.2f}%")

# full classification report
print("\nDetailed Report:")
print(classification_report(y_true, y_pred, target_names=labels))
