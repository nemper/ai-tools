import os
import pandas as pd
import re

# -------------------- Configuration --------------------

CSV_FILE = 'gdrive_names_and_urls.csv'      # Path to your CSV file
DOWNLOAD_DIR = 'gdrive_files'               # Directory where files are downloaded
LANGUAGE_CODES = {'en', 'sr', 'hr'}         # Supported language codes

# -------------------- Functions --------------------

def extract_base_name(file_name):
    """
    Extracts the base name of the file by removing the language suffix and extension.
    Example: 'Restorative_Solutions_en.pdf' -> 'Restorative_Solutions'
    """
    pattern = r'^(.*)_(' + '|'.join(LANGUAGE_CODES) + r')\.[^.]+$'
    match = re.match(pattern, file_name)
    if match:
        return match.group(1)
    return None

def find_en_duplicates(df):
    """
    Identifies '_en' files that have duplicates in other language codes.
    
    Returns a list of file names ending with '_en' to be deleted.
    """
    # Create a new column for base names (temporary, not saved to CSV)
    base_names = df['File Name'].apply(extract_base_name)
    
    # Create a new column for language codes (temporary, not saved to CSV)
    language_codes = df['File Name'].apply(
        lambda x: re.findall(r'_(' + '|'.join(LANGUAGE_CODES) + r')\.[^.]+$', x)[0]
        if re.findall(r'_(' + '|'.join(LANGUAGE_CODES) + r')\.[^.]+$', x) else None
    )
    
    # Combine base names and language codes into a temporary DataFrame
    temp_df = pd.DataFrame({
        'File Name': df['File Name'],
        'Base Name': base_names,
        'Language Code': language_codes
    })
    
    # Drop rows where base name or language code couldn't be extracted
    temp_df = temp_df.dropna(subset=['Base Name', 'Language Code'])
    
    # Group by base name and collect unique language codes
    grouped = temp_df.groupby('Base Name')['Language Code'].apply(set)
    
    # Identify base names with more than one language code
    duplicate_base_names = grouped[grouped.apply(lambda x: len(x) > 1)].index.tolist()
    
    # From these, select the '_en' files
    en_duplicates = temp_df[
        (temp_df['Base Name'].isin(duplicate_base_names)) &
        (temp_df['Language Code'] == 'en')
    ]['File Name'].tolist()
    
    return en_duplicates

def delete_files(file_names, download_dir):
    """
    Deletes specified files from the download directory.
    
    Args:
        file_names (list): List of file names to delete.
        download_dir (str): Directory where files are downloaded.
    """
    for file_name in file_names:
        file_path = os.path.join(download_dir, file_name)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            else:
                print(f"File not found, skipping deletion: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

def remove_en_duplicates(csv_file, download_dir):
    """
    Processes the CSV to remove '_en' duplicates and deletes them from the download directory.
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{csv_file}' is empty.")
        return
    except Exception as e:
        print(f"An error occurred while reading '{csv_file}': {e}")
        return

    # Ensure required columns exist
    if 'File Name' not in df.columns:
        print("Error: 'File Name' column is missing in the CSV.")
        return

    # Identify '_en' duplicates
    en_duplicates = find_en_duplicates(df)

    if not en_duplicates:
        print("No '_en' duplicate files found to delete.")
        return

    print(f"Found {len(en_duplicates)} '_en' duplicate files to delete.")

    # Delete files from download directory
    delete_files(en_duplicates, download_dir)

    # Remove '_en' entries from the DataFrame
    df_updated = df[~df['File Name'].isin(en_duplicates)]

    # Save the updated CSV with only original columns
    try:
        df_updated.to_csv(csv_file, index=False, encoding='utf-8-sig', columns=['File Name', 'URL'])
        print(f"Updated CSV '{csv_file}' saved successfully.")
    except Exception as e:
        print(f"An error occurred while saving the updated CSV: {e}")

if __name__ == "__main__":
    remove_en_duplicates(CSV_FILE, DOWNLOAD_DIR)
