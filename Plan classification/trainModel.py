from ultralytics import YOLO

# Build from YAML and load weights
model = YOLO("runs/classify/train5/weights/last.pt")

# Train the model
print("Starting training...")
results = model.train(data=r'C:\Users\Administrator\Documents\data\plan_classification\train', epochs=100, imgsz=1200, patience=10)
print("Training completed.")

# Save the model
model.save("plan_classifier_yolo.pt")
print("Model saved as 'plan_classifier_yolo.pt'.")
