from ultralytics import YOLO
import numpy as np

# Load the best weights from the previous run
model = YOLO("runs/classify/train2/weights/best.pt")

# Predict on new data
predictions = model.predict(
    source=r'E:\data\plan_classification\test',
)

# Write results to a file
with open("prediction_results.txt", "w") as f:
    for result in predictions:
        if result.probs is None:
            print(f"No probabilities for {result.path}")
            continue

        probs = result.probs.data

        # Get prediction
        predicted_index = np.array(probs).argmax()
        predicted_name = model.names[predicted_index]
        confidence = probs[predicted_index]

        print(f"Image: {result.path.split('\\')[-1]}   --------   Predicted class: {predicted_name} with probability {confidence:.2f}")
        f.write(f"Image: {result.path.split('\\')[-1]}   --------   Predicted class: {predicted_name} with probability {confidence:.2f}\n")

print("Prediction completed and results saved to 'prediction_results.txt'.")