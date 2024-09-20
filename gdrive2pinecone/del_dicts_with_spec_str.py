import os
import json

def remove_dicts_with_string(json_data, target_string):
    """
    This function takes a list of dictionaries (json_data) and removes
    any dictionaries that contain the target_string in any of their values.
    """
    cleaned_data = [
        item for item in json_data
        if all(target_string not in str(value) for value in item.values())
    ]
    return cleaned_data

def clean_json_files(folder_path, target_string):
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r') as file:
                    content = json.load(file)
                
                if isinstance(content, list) and all(isinstance(item, dict) for item in content):
                    cleaned_content = remove_dicts_with_string(content, target_string)
                    
                    # If changes were made, rewrite the JSON file
                    if cleaned_content != content:
                        print(f"Cleaning {filename}")
                        with open(file_path, 'w') as file:
                            json.dump(cleaned_content, file, indent=4)
            except json.JSONDecodeError:
                print(f"Skipping {filename} as it's not a valid JSON file")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

folder_path = os.path.join(os.getcwd(), "json_output")
target_string = "We reserve the right to make any alterations which may be required due to technical improvements."

clean_json_files(folder_path, target_string)

print("Cleaning completed.")
