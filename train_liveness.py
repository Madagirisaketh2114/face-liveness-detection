import cv2
import numpy as np
import tensorflow as tf
import os
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Set paths
dataset_path = r"C:\Users\Dhanush\OneDrive\Desktop\Face-Liveness-Detection\sample_liveness_data"
model_path = "liveness.model.keras"
label_encoder_path = "le.pickle"

# Load dataset
data = []
labels = []
categories = ["real", "fake"]

for category in categories:
    path = os.path.join(dataset_path, category)
    label = 1 if category == "real" else 0  # 1 = Real, 0 = Fake
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.resize(image, (64, 64))
        data.append(image)
        labels.append(label)

# Convert data to NumPy array and normalize
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

# Save label encoder
with open(label_encoder_path, "wb") as f:
    pickle.dump(categories, f)

# Data augmentation
datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2,
                             height_shift_range=0.2, shear_range=0.15, horizontal_flip=True)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")  # Binary classification (Real or Fake)
])

# Compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
model.fit(datagen.flow(data, labels, batch_size=32), epochs=10)

# Save model
model.save(model_path)
print(f"✅ Model saved to {model_path}")
