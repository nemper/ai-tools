import os
import json

def delete_empty_jsons(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r') as file:
                    content = json.load(file)
                
                if content == []:
                    print(f"Deleting {filename} as it contains an empty list")
                    os.remove(file_path)
            except json.JSONDecodeError:
                print(f"Skipping {filename} as it's not a valid JSON file")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

folder_path = os.path.join(os.getcwd(), "json_output")

delete_empty_jsons(folder_path)
