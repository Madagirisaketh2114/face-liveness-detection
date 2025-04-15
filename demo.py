import cv2
import numpy as np
import tensorflow as tf
import os

# Load trained model
model = tf.keras.models.load_model("liveness.model.keras")

# Path to the dataset
dataset_path = r"C:\Users\Dhanush\OneDrive\Desktop\Face-Liveness-Detection\sample_liveness_data"

# Loop through 'real' and 'fake' folders
for category in ["real", "fake"]:
    category_path = os.path.join(dataset_path, category)
    
    if not os.path.exists(category_path):
        print(f"Error: Path not found - {category_path}")
        continue

    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        
        # Check if the file is a valid image
        if not (img_name.lower().endswith(".png") or img_name.lower().endswith(".jpg") or img_name.lower().endswith(".jpeg")):
            print(f"Skipping non-image file: {img_name}")
            continue
        
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Image {img_name} could not be loaded.")
            continue

        # Preprocess image
        image_resized = cv2.resize(image, (64, 64))
        image_normalized = image_resized / 255.0
        input_image = np.expand_dims(image_normalized, axis=0)

        # Predict
        prediction = model.predict(input_image)
        label = "Real" if prediction > 0.5 else "Fake"

        print(f"Liveness Detection Result for {category}/{img_name}: {label}")
