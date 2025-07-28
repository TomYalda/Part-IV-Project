import re

# File paths
predicted_classes_file = "prediction_results.txt"
ground_truth_classes_file = "correct_classifications.txt"

def parse_prediction_file(text):
    """
    Parses lines like:
    Image: ADDITIONAL_PLANS_INFORMATION (1098600)_10.jpg   --------   Predicted class: structural with probability 0.91
    Returns: dict { filename: (class, confidence) }
    """
    predictions = {}
    for line in text.strip().splitlines():
        m = re.match(
            r"Image:\s+(.+?)\s+--------\s+Predicted class:\s+(\w+)\s+with probability\s+([\d.]+)",
            line
        )
        if m:
            filename, predicted_class, confidence = m.groups()
            predictions[filename.strip()] = (predicted_class.strip(), float(confidence))
    return predictions

# Read files
with open(predicted_classes_file, "r", encoding="utf-8") as f:
    predicted_file_contents = f.read()

with open(ground_truth_classes_file, "r", encoding="utf-8") as f:
    ground_truth_file_contents = f.read()

# Parse
predicted_classes = parse_prediction_file(predicted_file_contents)
ground_truth_classes = parse_prediction_file(ground_truth_file_contents)

# Compare results
total_differences = 0
print("Differences between predicted and ground truth classifications:")
for filename in sorted(set(predicted_classes) | set(ground_truth_classes)):
    predicted_label = predicted_classes.get(filename, (None, None))[0]
    ground_truth_label = ground_truth_classes.get(filename, (None, None))[0]
    
    if ground_truth_label == "Unknown":
        continue  # Skip if ground truth says Unknown

    if predicted_label != ground_truth_label:
        predicted_confidence = predicted_classes.get(filename, (None, None))[1]
        ground_truth_confidence = ground_truth_classes.get(filename, (None, None))[1]
        print(
            f"{filename}: predicted='{predicted_label}' ({predicted_confidence}) vs ground_truth='{ground_truth_label}' ({ground_truth_confidence})"
        )
        total_differences += 1

# Summary
print()
if total_differences == 0:
    print("No differences found between the predicted and ground truth classifications.")
else:
    print(f"\nTotal differences found: {total_differences}")
