import os
import json
import pandas as pd

from io import FileIO, StringIO
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------- Configuration --------------------

SCOPES = ['https://www.googleapis.com/auth/drive']
CREDENTIALS_FILE = 'denty.json'
FOLDER_IDS = [
    '17vXjaLOFp0iPg6WbnUlNBt5TxguazHx5',
    '18S-dIodtT6RQ6Josiw1BOqkw05TRAVkL',
    '1e12X3dkR7hlpV0n9OjLni4FGQQOxSzRl',
    '1dzglHgMpQDO6E2yyAdDHTbn1xmIY75mq'
]
DOWNLOAD_DIR = 'gdrive_files'
CSV_FILE = 'gdrive_names_and_urls.csv'

MAX_WORKERS = os.cpu_count() or 4

# -------------------- Functions --------------------

def get_service_account_info(credentials_file):
    """Load and return the service account information from a JSON file."""
    with open(credentials_file) as f:
        service_account_info = json.load(f)
    return service_account_info

def create_drive_service(service_account_info):
    """Create and return a Google Drive service instance."""
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info, scopes=SCOPES
    )
    service = build('drive', 'v3', credentials=credentials)
    return service

def list_drive_files(service, folder_id):
    """Recursively list all files in a folder and its subfolders."""
    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(
        q=query,
        pageSize=1000,
        fields="nextPageToken, files(id, name, webViewLink, mimeType)"
    ).execute()
    items = results.get('files', [])
    
    files = []
    for item in items:
        if item['mimeType'] == 'application/vnd.google-apps.folder':
            # Recursively fetch files from subfolders
            files.extend(list_drive_files(service, item['id']))
        else:
            files.append(item)
    
    return files

def process_multiple_folders(service, folder_ids):
    """List all files from multiple folders."""
    all_files = []
    for folder_id in folder_ids:
        print(f"Fetching files from folder ID: {folder_id}...")
        folder_files = list_drive_files(service, folder_id)
        all_files.extend(folder_files)
    return all_files

def download_file(service_account_info, file, download_dir):
    """
    Download a single file from Google Drive.
    
    Args:
        service_account_info (dict): Service account credentials.
        file (dict): File information dictionary containing 'id' and 'name'.
        download_dir (str): Directory to save the downloaded file.
    
    Returns:
        tuple: (file_name, success_flag, error_message)
    """
    file_id = file['id']
    file_name = file['name']
    request = None
    try:
        # Create a new service instance for each thread
        service = create_drive_service(service_account_info)
        request = service.files().get_media(fileId=file_id)
        
        file_path = os.path.join(download_dir, file_name)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with FileIO(file_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print(f"Downloading {file_name}: {int(status.progress() * 100)}%.")
        print(f"Downloaded {file_name} successfully!")
        return (file_name, True, "")
    except Exception as e:
        print(f"Failed to download {file_name}. Error: {e}")
        return (file_name, False, str(e))

def download_drive_files_concurrently(service_account_info, files, download_dir, max_workers=12):
    """Download all files to the specified directory using parallel processing."""
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    download_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_file = {
            executor.submit(download_file, service_account_info, file, download_dir): file
            for file in files
        }
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                file_name, success, error = future.result()
                if success:
                    download_results.append(file)
                else:
                    print(f"Error downloading {file_name}: {error}")
            except Exception as exc:
                print(f"Generated an exception: {exc}")
    
    return download_results

def generate_csv(files, csv_file):
    """Generate a CSV file with File Name and URL."""
    if not files:
        print('No files to write to CSV.')
        return
    
    df = pd.DataFrame(files)
    df = df[['name', 'webViewLink']]
    df.columns = ['File Name', 'URL']
    
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8-sig', lineterminator='\n')
    csv_data = csv_buffer.getvalue()
    
    with open(csv_file, 'w', encoding='utf-8-sig') as f:
        f.write(csv_data)
    
    print(f"CSV file '{csv_file}' generated successfully!")

# -------------------- Main Execution --------------------

def main():
    # Step 1: Load service account information
    service_account_info = get_service_account_info(CREDENTIALS_FILE)
    
    # Step 2: Create a service instance for listing files
    service = create_drive_service(service_account_info)
    
    # Step 3: List all files from multiple folders
    files = process_multiple_folders(service, FOLDER_IDS)
    
    if not files:
        print('No files found in the specified folders.')
        return
    
    print(f"Found {len(files)} files. Starting download...")
    
    # Step 4: Download all files concurrently
    downloaded_files = download_drive_files_concurrently(
        service_account_info, files, DOWNLOAD_DIR, max_workers=MAX_WORKERS
    )
    
    # Optional: Filter out failed downloads if needed
    successful_downloads = [file for file in downloaded_files]
    
    # Step 5: Generate CSV with File Name and URL
    generate_csv(successful_downloads, CSV_FILE)
    
    print('All files downloaded and CSV generated successfully.')

if __name__ == '__main__':
    main()
    