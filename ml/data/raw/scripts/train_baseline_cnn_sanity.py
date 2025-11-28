import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score

# ---------------- PATHS ----------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
LOG_DIR   = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.h5")
BEST_MODEL_PATH  = os.path.join(MODEL_DIR, "best_model.h5")
CSV_LOG_PATH     = os.path.join(LOG_DIR, "training_log.csv")

# ---------------- PARAMETERS ----------------
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 40
NUM_CLASSES = 7   # based on your dataset

# ---------------- DATA GENERATORS ----------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
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
print("\n⚖️ Class weights:", class_weights)

# ---------------- MODEL ----------------
model = Sequential([
    # Block 1
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1), padding='same'),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- CALLBACKS ----------------
csv_logger = CSVLogger(CSV_LOG_PATH, append=True)
checkpoint = ModelCheckpoint(BEST_MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

# ---------------- TRAIN ----------------
print("\n🚀 Starting full training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[csv_logger, checkpoint, early_stop, reduce_lr]
)

# ---------------- SAVE FINAL MODEL ----------------
model.save(FINAL_MODEL_PATH)
print(f"\n✅ Training complete.")
print(f"✅ Final model saved at: {FINAL_MODEL_PATH}")
print(f"✅ Best model saved at: {BEST_MODEL_PATH}")

# ---------------- EVALUATE ----------------
print("\n📊 Evaluating model on validation set...")
val_generator.reset()
y_pred_probs = model.predict(val_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_generator.classes
labels = list(val_generator.class_indices.keys())

print("\n🎯 Per-class accuracy:")
for i, label in enumerate(labels):
    idx = np.where(y_true == i)[0]
    acc = np.mean(y_pred[idx] == y_true[idx])
    print(f"{label}: {acc*100:.2f}%")

overall_acc = accuracy_score(y_true, y_pred)
print(f"\n⭐ Overall Validation Accuracy: {overall_acc*100:.2f}%")

print("\n📋 Detailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=labels))
