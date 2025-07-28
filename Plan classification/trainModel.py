from ultralytics import YOLO

# Build from YAML and load weights
model = YOLO("yolov8n-cls.pt")

# Train the model
print("Starting training...")
results = model.train(data=r'E:\data\plan_classification\train', epochs=100, imgsz=64)
print("Training completed.")

# Save the model
model.save("plan_classifier_yolo.pt")
print("Model saved as 'plan_classifier_yolo.pt'.")
