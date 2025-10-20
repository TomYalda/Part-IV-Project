import os

def normalize_filename(filename):
    """
    Normalizes a filename by removing its extension and stripping whitespace.
    This helps in matching filenames that may or may not have an extension
    across the two files.
    Example: 'document_1.jpg' -> 'document_1'
             'document_1' -> 'document_1'
    """
    # os.path.splitext splits 'file.ext' into ('file', '.ext')
    base_name, _ = os.path.splitext(filename)
    return base_name.strip()

def parse_full_text_file(filepath):
    """
    Parses the file with the long classification format (e.g., "Documents").
    
    Args:
        filepath (str): The path to the file.

    Returns:
        dict: A dictionary mapping normalized filenames to a tuple of 
              (simplified class, line number). Returns None if the file cannot be read.
    """
    classifications = {}
    # This map converts the full text classification to the single character format.
    class_map = {
        'Documents': 'D',
        'StructuralPlans': 'P'
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if ':' not in line:
                    continue
                
                parts = line.split(':', 1)
                filename = normalize_filename(parts[0])
                
                # The classification is the first word after the colon.
                details = parts[1].strip()
                classification_word = details.split(' ')[0]
                
                if classification_word in class_map:
                    classifications[filename] = (class_map[classification_word], i)
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return None
    
    return classifications

def parse_single_char_file(filepath):
    """
    Parses the file with the single character classification format ('D' or 'P').
    
    Args:
        filepath (str): The path to the file.

    Returns:
        dict: A dictionary mapping normalized filenames to a tuple of 
              (class, line number). Returns None if the file cannot be read.
    """
    classifications = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if ':' not in line:
                    continue
                
                parts = line.split(':', 1)
                filename = normalize_filename(parts[0])
                class_char = parts[1].strip().upper()
                
                if class_char in ['D', 'P']:
                    classifications[filename] = (class_char, i)
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return None
        
    return classifications

def main():
    """
    Main function to run the file comparison.
    """
 
    file1_path = 'classifiersDeterminations.txt'
    file2_path = 'MyDeterminations.txt'
    # -------------------

    print("--- Starting File Classification Comparison ---")
    print(f"File 1 (Full Text): '{file1_path}'")
    print(f"File 2 (Single Char): '{file2_path}'")
    
    # Parse both files into dictionaries
    file1_data = parse_full_text_file(file1_path)
    file2_data = parse_single_char_file(file2_path)
    
    # Exit if there was an error reading files
    if file1_data is None or file2_data is None:
        print("\nComparison aborted due to file reading errors.")
        return
        
    print(f"\nLoaded and processed {len(file1_data)} entries from File 1.")
    print(f"Loaded and processed {len(file2_data)} entries from File 2.")

    mismatches_d_vs_p = []
    mismatches_p_vs_d = []
    
    # Iterate through the data from the first file and compare with the second
    for filename, (class1, line1) in file1_data.items():
        if filename in file2_data:
            class2, line2 = file2_data[filename]
            
            # Check for 'Documents' (D) vs 'P' mismatch
            if class1 == 'D' and class2 == 'P':
                mismatches_d_vs_p.append((filename, line1, line2))
            
            # Check for 'StructuralPlans' (P) vs 'D' mismatch
            elif class1 == 'P' and class2 == 'D':
                mismatches_p_vs_d.append((filename, line1, line2))

    print("\n--- Comparison Results ---")
    total_mismatches = len(mismatches_d_vs_p) + len(mismatches_p_vs_d)

    if total_mismatches == 0:
        print("No classification mismatches found. The files are consistent.")
    else:
        print(f"Found {total_mismatches} total mismatches.")
        
        if mismatches_d_vs_p:
            print(f"\n[TYPE 1] Classified as 'Documents' in File 1 but 'P' in File 2 ({len(mismatches_d_vs_p)} files):")
            print("-" * 80)
            print(f"{'Filename':<50} {'File 1 Line':<15} {'File 2 Line':<15}")
            print("-" * 80)
            for fname, l1, l2 in sorted(mismatches_d_vs_p):
                print(f"{fname:<50} {l1:<15} {l2:<15}")
        
        if mismatches_p_vs_d:
            print(f"\n[TYPE 2] Classified as 'StructuralPlans' in File 1 but 'D' in File 2 ({len(mismatches_p_vs_d)} files):")
            print("-" * 80)
            print(f"{'Filename':<50} {'File 1 Line':<15} {'File 2 Line':<15}")
            print("-" * 80)
            for fname, l1, l2 in sorted(mismatches_p_vs_d):
                print(f"{fname:<50} {l1:<15} {l2:<15}")

if __name__ == "__main__":
    main()

