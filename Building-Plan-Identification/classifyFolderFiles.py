import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model = load_model('best_classifier.h5')

# Class names should match the order from the training generator
class_names = ['Documents', 'StructuralPlans']

# Directory containing test images
test_dir = 'data/testingData/manualValidationData'

# Prepare output file
output_file = 'classifiersDeterminations.txt' 
with open(output_file, 'w') as f:
    # Get all file paths and sort them by file size (ascending)
    files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
    files_sorted = sorted(files, key=lambda x: os.path.getsize(os.path.join(test_dir, x)))
    for img_name in files_sorted:
        img_path = os.path.join(test_dir, img_name)
        try:
            # Load and preprocess the image
            img = image.load_img(img_path, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize same as training

            # Predict the class
            prediction = model.predict(img_array, verbose=0)
            pred_prob = prediction[0][0]
            predicted_class = class_names[int(pred_prob > 0.5)]

            # Write result to file with confidence
            f.write(f"{img_name}: {predicted_class} ({pred_prob * 100:.2f}% confidence)\n")
        except Exception as e:
            f.write(f"{img_name}: Error processing image ({str(e)})\n")

print(f"Predictions written to {output_file}")