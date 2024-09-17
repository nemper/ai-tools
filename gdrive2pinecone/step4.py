import os
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

json_folder_path = './json_output'
csv_file_path = 'google_drive_files.csv'

csv_df = pd.read_csv(csv_file_path, header=None, names=["source", "url"])

url_dict = {source.lower(): url for source, url in zip(csv_df["source"], csv_df["url"])}


def process_json_file(filename):
    if filename.endswith(".json"):
        json_file_path = os.path.join(json_folder_path, filename)
        print(f"Processing file: {filename}")

        # Open the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        # Track if any URLs were updated
        updated = False

        # Update the 'url' key in each dict
        for entry in data:
            source = entry.get("source")
            if source and source.lower() in url_dict:
                old_url = entry.get("url")
                new_url = url_dict[source.lower()]
                if old_url != new_url:
                    print(f"Updating URL for source: {source}")
                    entry["url"] = new_url
                    updated = True

        # Save the updated JSON file only if changes were made
        if updated:
            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, ensure_ascii=False, indent=4)
            print(f"File updated: {filename}")
        else:
            print(f"No updates needed for: {filename}")

# Get list of all JSON files in the folder
json_files = [f for f in os.listdir(json_folder_path) if f.endswith(".json")]

# Use ThreadPoolExecutor to process files in parallel with 4 workers
with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(process_json_file, json_files)

print("Processing complete.")
