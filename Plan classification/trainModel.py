import os

from ultralytics import YOLO

# Build from YAML and load weights
model = YOLO("yolov8l-cls.pt")


# Train the model
print("Starting training...")
results = model.train(data="data/plan_classification/split-data", epochs=100, imgsz=600, patience=10, val=True)
print("Training completed.")

# Save the model
model.save("plan_classifier_yolo.pt")
print("Model saved as 'plan_classifier_yolo.pt'.")
