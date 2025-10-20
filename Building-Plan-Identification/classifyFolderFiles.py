import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import time

# Load the trained model
model = load_model('classifier.h5')

# Class names should match the order from the training generator
class_names = ['Documents', 'StructuralPlans']

# Directory containing test images
test_dir = 'data/test_images'

# Prepare output file
output_file = 'classifiersDeterminations.txt'
files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
files_sorted = sorted(files, key=lambda x: os.path.getsize(os.path.join(test_dir, x)))

# Time only the prediction step
start_time = time.time()
predictions = []
for img_name in files_sorted:
    img_path = os.path.join(test_dir, img_name)
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(900, 900))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize same as training

        # Predict the class
        prediction = model.predict(img_array, verbose=0)
        pred_prob = prediction[0][0]
        predicted_class = class_names[int(pred_prob > 0.5)]
        predictions.append((img_name, predicted_class, pred_prob))
    except Exception as e:
        predictions.append((img_name, None, str(e)))
end_time = time.time()
elapsed = end_time - start_time

# Write results to a file (not timed)
with open(output_file, 'w') as f:
    for item in predictions:
        if item[1] is not None:
            f.write(f"{item[0]}: {item[1]} ({float(item[2]) * 100:.2f}% confidence)\n")
        else:
            f.write(f"{item[0]}: Error processing image ({item[2]})\n")
print(f"Predictions written to {output_file}")
print(f"Classification took {elapsed:.2f} seconds (prediction only).")