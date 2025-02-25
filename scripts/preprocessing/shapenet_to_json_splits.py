import os
import json
import random

def export_subdirectories_to_json(parent_directory, class_name, train_file, test_file, all_file, train_split):
    """
    Exports the names of all subdirectories within the specified parent directory
    to three JSON files: train, test, and all.

    :param parent_directory: The path to the parent directory
    :param train_file: The path to the output JSON file for the training split
    :param test_file: The path to the output JSON file for the testing split
    :param all_file: The path to the output JSON file for all subdirectories
    :param train_split: The percentage of subdirectories to include in the training split (0-100)
    """
    try:
        # Get a list of all subdirectories within the parent directory
        subdirectories = [
            name for name in os.listdir(parent_directory)
            if os.path.isdir(os.path.join(parent_directory, name))
        ]

        # Shuffle the list to ensure randomness
        random.shuffle(subdirectories)

        # Calculate the split index
        split_index = int(len(subdirectories) * (train_split / 100.0))

        # Split the subdirectories into train and test
        train_subdirectories = subdirectories[:split_index]
        test_subdirectories = subdirectories[split_index:]

        # Create the JSON structures
        train_data = {
            "ShapeNetV2": {
                class_name: train_subdirectories
            }
        }

        test_data = {
            "ShapeNetV2": {
                class_name: test_subdirectories
            }
        }

        all_data = {
            "ShapeNetV2": {
                class_name: subdirectories
            }
        }

        # Write the data to JSON files
        with open(train_file, "w") as train_json_file:
            json.dump(train_data, train_json_file, indent=4)

        with open(test_file, "w") as test_json_file:
            json.dump(test_data, test_json_file, indent=4)

        with open(all_file, "w") as all_json_file:
            json.dump(all_data, all_json_file, indent=4)

        print(f"Train-test split and all data exported successfully: \nTrain: {train_file}\nTest: {test_file}\nAll: {all_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    parent_directory = input("Enter the path to the shapenet dataset: ")
    class_name = input("Enter the class name: ")
    train_file = input("Enter the path to the output JSON file for the training split: ")
    test_file = input("Enter the path to the output JSON file for the testing split: ")
    all_file = input("Enter the path to the output JSON file for all subdirectories: ")
    train_split = float(input("Enter the percentage of data for the training split (0-100): "))

    export_subdirectories_to_json(parent_directory+"/"+class_name, class_name, train_file, test_file, all_file, train_split)
