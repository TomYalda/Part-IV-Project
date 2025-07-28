import re

# Update these paths to your actual file locations
with_confidence_path = "classifiersDeterminations.txt"
without_confidence_path = "MyDetermination.txt"

def parse_results_with_confidence(text):
    d = {}
    for line in text.strip().splitlines():
        m = re.match(r"(.+?):\s+(\w+)\s+\(([\d.]+)% confidence\)", line)
        if m:
            fname, label, conf = m.groups()
            d[fname.strip()] = (label.strip(), float(conf))
        else:
            m = re.match(r"(.+?):\s+(\w+)", line)
            if m:
                fname, label = m.groups()
                d[fname.strip()] = (label.strip(), None)
    return d

def parse_results_without_confidence(text):
    d = {}
    for line in text.strip().splitlines():
        m = re.match(r"(.+?):\s+(\w+)", line)
        if m:
            fname, label = m.groups()
            d[fname.strip()] = label.strip()
    return d

# Read the files
with open(with_confidence_path, "r", encoding="utf-8") as f:
    results_with_confidence = f.read()

with open(without_confidence_path, "r", encoding="utf-8") as f:
    results_without_confidence = f.read()

with_conf = parse_results_with_confidence(results_with_confidence)
without_conf = parse_results_without_confidence(results_without_confidence)

print("Differences between the two result sets:")
for fname in sorted(set(with_conf) | set(without_conf)):
    label1 = with_conf.get(fname, (None, None))[0]
    label2 = without_conf.get(fname)
    if label2 == "Unknown":
        continue  # Ignore entries where without_conf is Unknown
    if label1 != label2:
        conf = with_conf.get(fname, (None, None))[1]
        print(f"{fname}: with_conf='{label1}' ({conf}%) vs without_conf='{label2}'")