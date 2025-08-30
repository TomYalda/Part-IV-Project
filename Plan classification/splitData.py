import os
import random
import shutil
from pathlib import Path

# paths
dataset_dir = Path(r"C:\Users\Administrator\Documents\data\plan_classification\train")
output_dir = Path(r"C:\Users\Administrator\Documents\data\plan_classification\split-data")

# split ratios
train_split = 0.7
val_split = 0.2  # remaining goes to test

# class folders (each subfolder = a class)
classes = [d for d in os.listdir(dataset_dir) if (dataset_dir / d).is_dir()]

for split in ["train", "val"]:
    for cls in classes:
        (output_dir / split / cls).mkdir(parents=True, exist_ok=True)


# test folder with class subfolders
for cls in classes:
    (output_dir / "test" / cls).mkdir(parents=True, exist_ok=True)

for cls in classes:
    images = list((dataset_dir / cls).glob("*.*"))
    random.shuffle(images)

    n = len(images)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in splits.items():
        for f in files:
            if split == "test":
                # copy into class subfolders for test
                shutil.copy(f, output_dir / split / cls / f.name)
            else:
                # keep class subfolders for train/val
                shutil.copy(f, output_dir / split / cls)

# Generate correct_classifications.txt for test set
correct_classifications_path = Path(r"c:\Users\Administrator\Documents\Classification Models\Part-IV-Project\Plan classification\correct_classifications.txt")
with open(correct_classifications_path, "w") as out_file:
    test_dir = output_dir / "test"
    for cls in classes:
        class_dir = test_dir / cls
        for img_file in class_dir.glob("*.*"):
            out_file.write(f"Image: {img_file.name}   --------   Predicted class: {cls} with probability 1.00\n")

# Move all test images out of subfolders into flat test folder
for cls in classes:
    class_dir = output_dir / "test" / cls
    for img_file in class_dir.glob("*.*"):
        shutil.move(str(img_file), str(output_dir / "test" / img_file.name))
    # Remove the now-empty class subfolder
    os.rmdir(class_dir)
