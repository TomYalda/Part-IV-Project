import re
import csv
from sklearn.metrics import accuracy_score, classification_report

with_confidence_path = "classifiersDeterminations.txt"
without_confidence_path = "MyDeterminations.txt"
output_csv = "classification_metrics.csv"

def extract_labels_with_confidence(text):
    """Extract just the class label from lines with confidence file."""
    labels = []
    pattern = re.compile(r":\s+(\w+)\s+\([\d.]+% confidence\)")
    for line in text.strip().splitlines():
        m = pattern.search(line)
        if m:
            labels.append(m.group(1))
    return labels

def extract_labels_without_confidence(text):
    """Extract just the class label from lines in ground truth file."""
    labels = []
    pattern = re.compile(r":\s+(\w+)$")
    for line in text.strip().splitlines():
        m = pattern.search(line)
        if m:
            labels.append(m.group(1))
    return labels

# Read both files
with open(with_confidence_path, "r", encoding="utf-8") as f:
    with_conf_text = f.read()
with open(without_confidence_path, "r", encoding="utf-8") as f:
    without_conf_text = f.read()

# Extract labels
labels_with_conf = extract_labels_with_confidence(with_conf_text)
labels_without_conf = extract_labels_without_confidence(without_conf_text)

# Make sure lengths match
min_len = min(len(labels_with_conf), len(labels_without_conf))
labels_with_conf = labels_with_conf[:min_len]
labels_without_conf = labels_without_conf[:min_len]

# Filter out Unknowns in ground truth
y_true, y_pred = [], []
for pred, gt in zip(labels_with_conf, labels_without_conf):
    if gt == "Unknown":
        continue
    y_true.append(gt)
    y_pred.append(pred)

# Print mismatches
print("\n🔍 Differences found:")
for i, (pred, gt) in enumerate(zip(y_pred, y_true), start=1):
    if pred != gt:
        print(f"Line {i}: with_conf='{pred}' vs without_conf='{gt}'")

# Compute metrics
all_labels = ["Documents", "StructuralPlans"]
accuracy = accuracy_score(y_true, y_pred)
report_dict = classification_report(
    y_true, y_pred, labels=all_labels, output_dict=True, zero_division=0
)

# Save report to CSV
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Class", "Precision", "Recall", "F1-score", "Support"])
    for cls in all_labels + ["accuracy", "macro avg", "weighted avg"]:
        if cls == "accuracy":
            writer.writerow([cls, "", "", f"{accuracy:.4f}", len(y_true)])
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
print(f"\n✅ Results saved to {output_csv}")
