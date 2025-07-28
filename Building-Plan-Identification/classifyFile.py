import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model = load_model('earlyStopClassifier.h5')

# Class names should match the order from the training generator
class_names = ['Documents', 'StructuralPlans']

# Define image file path here (change this to your image)
img_path = 'data/testingData/manualValidationData/1285-Combined_229.jpg'

# Validate file existence
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found at {img_path}")

# Load and preprocess the image
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize same as training

# Predict the class
prediction = model.predict(img_array)

# Interpret prediction for binary classification
pred_prob = prediction[0][0]
predicted_class = class_names[int(pred_prob > 0.5)]

# Show result with confidence
print(f"Prediction for {img_path}: {predicted_class} ({pred_prob * 100:.2f}% confidence)")
