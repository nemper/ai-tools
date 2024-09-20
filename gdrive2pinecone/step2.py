import os
import json
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ['https://www.googleapis.com/auth/drive']
CREDENTIALS_FILE = 'denty.json'
FOLDER_ID = '1IIJ27nxhK_id93sJ8YhMwe3xTRkawB8S'
DOWNLOAD_DIR = 'out'

def get_drive_service():
    with open(CREDENTIALS_FILE) as f:
        service_account_info = json.load(f)
    
    credentials = service_account.Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    service = build('drive', 'v3', credentials=credentials)
    return service

def list_drive_files(service, folder_id):
    """Recursively list all files in the folder and subfolders."""
    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(q=query, pageSize=1000, fields="nextPageToken, files(id, name, mimeType)").execute()
    items = results.get('files', [])
    
    files = []
    for item in items:
        if item['mimeType'] == 'application/vnd.google-apps.folder':
            # Recursively fetch files from subfolders
            files.extend(list_drive_files(service, item['id']))
        else:
            files.append(item)
    
    return files

def download_file(service, file_id, file_name, download_dir):
    """Download a single file from Google Drive."""
    request = service.files().get_media(fileId=file_id)
    file_path = os.path.join(download_dir, file_name)
    
    with io.FileIO(file_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Downloading {file_name} {int(status.progress() * 100)}%.")

def download_drive_files(service, files, download_dir):
    """Download all files to the specified directory."""
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    for file in files:
        print(f"Downloading {file['name']}...")
        download_file(service, file['id'], file['name'], download_dir)
        print(f"{file['name']} downloaded successfully!")

def main():
    service = get_drive_service()
    files = list_drive_files(service, FOLDER_ID)
    
    if not files:
        print('No files found.')
    else:
        print(f"Found {len(files)} files. Starting download...")
        download_drive_files(service, files, DOWNLOAD_DIR)
        print('All files downloaded successfully.')

if __name__ == '__main__':
    main()
