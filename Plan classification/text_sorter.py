import sys

# Sort each line in a text file alphabetically
def sort_text_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Strip whitespace and sort lines
    sorted_lines = sorted(line.strip() for line in lines)
    
    with open(file_path, 'w') as file:
        for line in sorted_lines:
            file.write(line + '\n')

if __name__ == "__main__":
    # Example usage
    file_path = sys.argv[1]  # Replace with your file path
    sort_text_file(file_path)
    print(f"Sorted lines in {file_path} alphabetically.")