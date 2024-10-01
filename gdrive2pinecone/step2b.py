import os
import pandas as pd
import json

# -------------------- Configuration --------------------

CSV_FILE = 'gdrive_names_and_urls.csv'      # Path to your CSV file
JSON_FOLDER = 'gdrive_jsons'                # Directory containing JSON files
DOWNLOAD_DIR = 'gdrive_files'               # Directory where files are downloaded (for reference)
LANGUAGE_CODES = {'en', 'sr', 'hr'}         # Supported language codes (if needed)

# -------------------- Functions --------------------

def load_csv_to_dict(csv_file):
    """
    Loads the CSV file into a dictionary for quick lookup.
    
    Args:
        csv_file (str): Path to the CSV file.
    
    Returns:
        dict: Dictionary with 'File Name' as keys and 'URL' as values.
    """
    try:
        df = pd.read_csv(csv_file)
        if 'File Name' not in df.columns or 'URL' not in df.columns:
            print(f"Error: CSV file '{csv_file}' must contain 'File Name' and 'URL' columns.")
            return {}
        # Create a dictionary for lookup
        url_dict = pd.Series(df.URL.values, index=df['File Name']).to_dict()
        return url_dict
    except FileNotFoundError:
        print(f"Error: The CSV file '{csv_file}' was not found.")
        return {}
    except pd.errors.EmptyDataError:
        print(f"Error: The CSV file '{csv_file}' is empty.")
        return {}
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return {}

def update_json_file(json_path, url_dict):
    """
    Updates the 'url' field in each dictionary within the JSON file based on the 'source' field.
    
    Args:
        json_path (str): Path to the JSON file.
        url_dict (dict): Dictionary with 'File Name' as keys and 'URL' as values.
    
    Returns:
        int: Number of URLs updated.
        int: Number of sources not found in the CSV.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        updated_count = 0
        not_found_count = 0
        
        for entry in data:
            source = entry.get('source', '').strip()
            if source in url_dict:
                if entry.get('url', '') != url_dict[source]:
                    entry['url'] = url_dict[source]
                    updated_count += 1
            else:
                # Source not found in CSV
                not_found_count += 1
                # Optionally, you can log or handle this case as needed
                # For example, you might set 'url' to None or leave it empty
                # entry['url'] = None  # Uncomment if you want to clear the URL
                print(f"Warning: Source '{source}' not found in CSV. URL not updated.")
        
        # Save the updated JSON back to the file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        return updated_count, not_found_count
    
    except FileNotFoundError:
        print(f"Error: The JSON file '{json_path}' was not found.")
        return 0, 0
    except json.JSONDecodeError:
        print(f"Error: The JSON file '{json_path}' is not properly formatted.")
        return 0, 0
    except Exception as e:
        print(f"An error occurred while processing '{json_path}': {e}")
        return 0, 0

def process_all_json_files(json_folder, url_dict):
    """
    Processes all JSON files in the specified folder.
    
    Args:
        json_folder (str): Path to the folder containing JSON files.
        url_dict (dict): Dictionary with 'File Name' as keys and 'URL' as values.
    
    Returns:
        dict: Summary of updates per file.
    """
    summary = {}
    if not os.path.isdir(json_folder):
        print(f"Error: The folder '{json_folder}' does not exist.")
        return summary
    
    json_files = [file for file in os.listdir(json_folder) if file.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in folder '{json_folder}'.")
        return summary
    
    for json_file in json_files:
        json_path = os.path.join(json_folder, json_file)
        print(f"Processing '{json_file}'...")
        updated, not_found = update_json_file(json_path, url_dict)
        summary[json_file] = {'Updated URLs': updated, 'Sources Not Found': not_found}
        print(f" - Updated URLs: {updated}")
        print(f" - Sources not found: {not_found}")
    
    return summary

# -------------------- Main Execution --------------------

def main():
    # Step 1: Load the CSV into a dictionary
    url_dict = load_csv_to_dict(CSV_FILE)
    if not url_dict:
        print("URL dictionary is empty. Exiting.")
        return
    
    # Step 2: Process all JSON files in the specified folder
    summary = process_all_json_files(JSON_FOLDER, url_dict)
    
    # Step 3: Print a summary of the updates
    if summary:
        print("\nSummary of Updates:")
        for json_file, stats in summary.items():
            print(f" - {json_file}: {stats['Updated URLs']} URLs updated, {stats['Sources Not Found']} sources not found.")
    else:
        print("No updates were made.")

if __name__ == "__main__":
    main()
