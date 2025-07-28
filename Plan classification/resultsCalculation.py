import csv
import re
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report

def parse_prediction_style_file(file_path):
    data = {}
    # This regex captures: filename and class
    pattern = re.compile(r"Image:\s+(.*?)\s+--------\s+Predicted class:\s+([a-zA-Z_]+)", re.IGNORECASE)

    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                filename = match.group(1).strip()
                label = match.group(2).strip().lower()
                data[filename] = label
            else:
                print(f"⚠️ Could not parse line: {line.strip()}")
    return data

def compute_multiclass_metrics(ground_truth_path, predictions_path, output_csv_path):
    ground_truth = parse_prediction_style_file(ground_truth_path)
    predictions = parse_prediction_style_file(predictions_path)

    # Automatically detect all unique classes in ground truth and predictions
    all_labels = sorted(list(set(ground_truth.values()) | set(predictions.values())))

    y_true = []
    y_pred = []

    for filename in ground_truth:
        if filename in predictions:
            y_true.append(ground_truth[filename])
            y_pred.append(predictions[filename])
        else:
            print(f"⚠️ Missing prediction for: {filename}")

    if not y_true:
        print("🚫 No valid matched entries between files.")
        return

    # Metrics
    precision = precision_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # Write to CSV
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Class', 'Precision', 'Recall'])
        for cls, p, r in zip(all_labels, precision, recall):
            writer.writerow([cls, f"{p:.4f}", f"{r:.4f}"])
        writer.writerow([])
        writer.writerow(['Overall Accuracy', f"{accuracy:.4f}"])

    # Print report
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, labels=all_labels, zero_division=0))

# ====== USAGE ======
ground_truth_file = 'correct_classifications.txt'
predictions_file = 'prediction_results.txt'
output_csv = 'classification_metrics_initial_model.csv'

compute_multiclass_metrics(ground_truth_file, predictions_file, output_csv)
