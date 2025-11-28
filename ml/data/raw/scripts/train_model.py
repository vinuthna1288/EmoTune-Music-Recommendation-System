import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

# -----------------------------
#  Adjust path to project root
# -----------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# -----------------------------
#  Configuration
# -----------------------------
INPUT_SHAPE = (48, 48, 1)       # grayscale images
BATCH_SIZE = 64
EPOCHS = 50
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")

# -----------------------------
#  Dataset paths (corrected)
# -----------------------------
# Go one level up from 'scripts' to 'raw'
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(BASE_DIR, "archive (2)", "train")
VAL_DIR = os.path.join(BASE_DIR, "archive (2)", "val")


# Debug print — see where code is looking
print("🔍 Current working directory:", os.getcwd())
print("📂 Training data folder:", TRAIN_DIR)
print("📂 Validation data folder:", VAL_DIR)

# Check if folders exist
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"❌ Training folder not found: {TRAIN_DIR}")
if not os.path.exists(VAL_DIR):
    raise FileNotFoundError(f"❌ Validation folder not found: {VAL_DIR}")

# -----------------------------
#  Create directories if needed
# -----------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------------------
#  Data Generators
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=INPUT_SHAPE[:2],
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=INPUT_SHAPE[:2],
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# -----------------------------
#  Model Definition
# -----------------------------
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# -----------------------------
#  Compilation
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
#  Callbacks
# -----------------------------
checkpoint_path = os.path.join(MODEL_DIR, "best_model.h5")
csv_log_path = os.path.join(LOG_DIR, "training_log.csv")

callbacks = [
    ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
    CSVLogger(csv_log_path)
]

# -----------------------------
#  Train Model
# -----------------------------
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)

# -----------------------------
#  Save final model
# -----------------------------
final_model_path = os.path.join(MODEL_DIR, "final_model.h5")
model.save(final_model_path)
print(f"\n✅ Model training complete. Saved at: {final_model_path}")
