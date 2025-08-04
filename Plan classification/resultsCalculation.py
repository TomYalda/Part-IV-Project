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

    # Automatically detect all unique classes
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

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Classification report (as dictionary)
    report_dict = classification_report(
        y_true, y_pred, labels=all_labels, output_dict=True, zero_division=0
    )

    # Write full classification report to CSV
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Class', 'Precision', 'Recall', 'F1-score', 'Support'])

        for cls in all_labels + ['accuracy', 'macro avg', 'weighted avg']:
            if cls == 'accuracy':
                writer.writerow([cls, '', '', f"{accuracy:.4f}", f"{len(y_true)}"])
            else:
                row = report_dict.get(cls, {})
                writer.writerow([
                    cls,
                    f"{row.get('precision', 0):.4f}",
                    f"{row.get('recall', 0):.4f}",
                    f"{row.get('f1-score', 0):.4f}",
                    int(row.get('support', 0))
                ])

    # Print to console
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, labels=all_labels, zero_division=0))

# ====== USAGE ======
ground_truth_file = 'correct_classifications.txt'
predictions_file = 'prediction_results.txt'
output_csv = 'classification_metrics_initial_model.csv'

compute_multiclass_metrics(ground_truth_file, predictions_file, output_csv)
