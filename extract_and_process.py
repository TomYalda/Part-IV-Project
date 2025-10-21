import os
import shutil
import zipfile
import tempfile
import time
import re
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image

# === INPUTS ===
ZIP_DIR = r"C:\Users\tomya\OneDrive - The University of Auckland\Documents\2025 Work\University\Part IV Project\AllFiles\zip_files"
OUTPUT_DIR = r"C:\Users\tomya\OneDrive - The University of Auckland\Documents\2025 Work\University\Part IV Project\AllFiles\images"
MAX_DEPTH = 10

# Ensure output and temp folder exists
os.makedirs("tmp", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_filename(s):
    """Sanitize filename by removing or replacing problematic characters."""
    s = re.sub(r"[^\w\s\-_]", "_", s)       # Replace non-safe chars with underscore
    s = re.sub(r"\s+", " ", s).strip()      # Normalize whitespace
    return s

def safe_filename(base, used_names):
    """Ensure filename is unique in output folder and not too long."""
    name_base = clean_filename(os.path.splitext(base)[0])[:100]  # Limit length
    name = f"{name_base}.jpeg"
    counter = 1
    while name in used_names:
        name = f"{name_base}_{counter}.jpeg"
        counter += 1
    used_names.add(name)
    return name

def convert_pdf_high_quality(pdf_path, output_folder, used_names):
    start = time.time()
    try:
        print(f"[PDF] Converting: {pdf_path}")
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        info = pdfinfo_from_path(pdf_path)
        num_pages = info["Pages"]

        for i in range(1, num_pages + 1):
            images = convert_from_path(pdf_path, dpi=300, first_page=i, last_page=i)
            img = images[0]
            raw_name = f"{base_name}_page_{i}.jpeg"
            output_name = safe_filename(raw_name, used_names)
            output_path = os.path.join(output_folder, output_name)
            img.convert("RGB").save(output_path, "JPEG", quality=95)
            print(f"    [PAGE {i}] Saved: {output_name}")

        print(f"[PDF] Done: {pdf_path} — {num_pages} pages in {time.time() - start:.2f}s\n")
    except Exception as e:
        print(f"[ERROR] PDF {pdf_path}: {e}\n")

def convert_image_to_jpeg(img_path, output_folder, used_names, label):
    start = time.time()
    try:
        print(f"[{label}] Converting: {img_path}")
        with Image.open(img_path) as img:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            raw_name = f"{base_name}.jpeg"
            output_name = safe_filename(raw_name, used_names)
            output_path = os.path.join(output_folder, output_name)
            img.convert("RGB").save(output_path, "JPEG", quality=95)
            print(f"    [SAVED] {output_name} in {time.time() - start:.2f}s\n")
    except Exception as e:
        print(f"[ERROR] {label} {img_path}: {e}\n")

def copy_jpeg(jpeg_path, output_folder, used_names):
    try:
        print(f"[JPEG] Copying: {jpeg_path}")
        base_name = os.path.basename(jpeg_path)
        output_name = safe_filename(base_name, used_names)
        output_path = os.path.join(output_folder, output_name)
        shutil.copy2(jpeg_path, output_path)
        print(f"    [COPIED] {output_name}\n")
    except Exception as e:
        print(f"[ERROR] JPEG {jpeg_path}: {e}\n")

def process_flat(root_folder, output_folder, depth=0):
    used_names = set()
    prefix = "    " * depth

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            ext = os.path.splitext(filename)[1].lower()

            print(f"{prefix}[INFO] Processing: {filepath}")
            if ext == ".pdf":
                convert_pdf_high_quality(filepath, output_folder, used_names)
            elif ext == ".png":
                convert_image_to_jpeg(filepath, output_folder, used_names, "PNG")
            elif ext in [".tif", ".tiff"]:
                convert_image_to_jpeg(filepath, output_folder, used_names, "TIFF")
            elif ext in [".jpg", ".jpeg"]:
                copy_jpeg(filepath, output_folder, used_names)
            elif ext == ".zip":
                extract_and_process(filepath, output_folder, depth=depth)
            else:
                print(f"{prefix}[SKIP] Unsupported file type: {filepath}\n")


def extract_and_process(zip_path, output_folder, depth=0):
    if depth > MAX_DEPTH:
        print(f"[WARNING] Maximum extraction depth {MAX_DEPTH} reached at {zip_path}. Skipping further extraction.\n")
        return

    prefix = "    " * depth
    print(f"{prefix}[INFO] Extracting '{zip_path}' to temp folder...")

    with tempfile.TemporaryDirectory(dir="tmp") as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.infolist():
                # Build full path
                member_path = os.path.join(temp_dir, member.filename)
                # Make sure parent dirs exist
                member_dir = os.path.dirname(member_path)
                os.makedirs(member_dir, exist_ok=True)
                # Extract file
                if not member.is_dir():
                    with zip_ref.open(member) as source, open(member_path, "wb") as target:
                        shutil.copyfileobj(source, target)

        print(f"{prefix}[INFO] Extraction complete.\n")
        print(f"{prefix}[INFO] Starting file processing...\n")
        process_flat(temp_dir, output_folder, depth=depth+1)

    print(f"{prefix}[DONE] Processed contents of: {zip_path}")


# === Run it ===
if __name__ == "__main__":
    for item in os.listdir(ZIP_DIR):
        item_path = os.path.join(ZIP_DIR, item)
        if os.path.isfile(item_path) and item.lower().endswith(".zip"):
            print(f"[MASTER] Starting to process ZIP: {item_path}")
            extract_and_process(item_path, OUTPUT_DIR)
        else:
            print(f"[MASTER] Skipping non-zip file: {item_path}")
