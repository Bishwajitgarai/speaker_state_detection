import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define dataset folder
DATASET_FOLDER = "output_images"

# Load images dynamically from folder
IMG_SIZE = (128, 128)  # Increase size for more detail
X, y = [], []

for label in ["speaking", "not_speaking"]:
    folder_path = os.path.join(DATASET_FOLDER, label)
    if not os.path.exists(folder_path):
        continue
    
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, IMG_SIZE)
            img = img.astype(np.float32) / 255.0  # Normalize
            X.append(img)
            y.append(0 if label == "not_speaking" else 1)  # Convert to binary labels

X = np.array(X)
y = np.array(y)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    fill_mode="nearest"
)
datagen.fit(X)  # Apply augmentation

# Build Improved CNN Model
model = keras.Sequential([
    layers.Conv2D(64, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),  # Reduce overfitting
    layers.Dense(1, activation="sigmoid")  # Binary classification
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003),
              loss="binary_crossentropy",
              metrics=["accuracy"])

# Train Model with Augmentation
model.fit(datagen.flow(X, y, batch_size=4), epochs=1, verbose=1)  # Increase epochs

# Save Model
model.save("speaking_detection_model.keras")

print("✅ Model saved as speaking_detection_model.keras")
