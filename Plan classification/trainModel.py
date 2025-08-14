import torch
import os

torch.set_num_threads(16)
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

from ultralytics import YOLO

# Build from YAML and load weights
model = YOLO("runs/classify/train8/weights/last.pt")

# Train the model
print("Starting training...")
results = model.train(data=r'C:\Users\Administrator\Documents\data\plan_classification\train', epochs=100, imgsz=1200, patience=10, resume=True, workers=8)
print("Training completed.")

# Save the model
model.save("plan_classifier_yolo.pt")
print("Model saved as 'plan_classifier_yolo.pt'.")
