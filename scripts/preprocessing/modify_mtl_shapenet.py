import os

# Function to process and modify .mtl files
def process_mtl_file(file_path):
    # Get the parent directory of the file
    parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(file_path)))

    try:
        # Read the contents of the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Replace ".." with the parent directory
        updated_lines = [line.replace("..", parent_dir) for line in lines]

        # Write the updated content back to the file
        with open(file_path, 'w') as file:
            file.writelines(updated_lines)

        print(f"Updated: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Function to search for .mtl files and process them
def search_and_replace(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.mtl'):
                file_path = os.path.join(root, file)
                process_mtl_file(file_path)

# Main function
def main():
    directory = input("Enter the directory to search: ").strip()
    if not os.path.isdir(directory):
        print("The specified path is not a directory.")
        return

    search_and_replace(directory)

if __name__ == "__main__":
    main()