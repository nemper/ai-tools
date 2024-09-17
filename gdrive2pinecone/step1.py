import json
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
import pandas as pd
from io import StringIO

SCOPES = ['https://www.googleapis.com/auth/drive']
CREDENTIALS_FILE = '...'
FOLDER_ID = '...'

def get_drive_service():
    # Read the credentials JSON from the file
    with open(CREDENTIALS_FILE) as f:
        service_account_info = json.load(f)
    
    # Create credentials from the service account information
    credentials = service_account.Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    service = build('drive', 'v3', credentials=credentials)
    return service

def list_drive_files(service, folder_id):
    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(q=query, pageSize=1000, fields="nextPageToken, files(id, name, webViewLink, mimeType)").execute()
    items = results.get('files', [])
    
    files = []
    for item in items:
        if item['mimeType'] == 'application/vnd.google-apps.folder':
            files.extend(list_drive_files(service, item['id']))
        else:
            files.append(item)
    
    return files

st.title('Google Drive File Viewer')

if st.button('List Google Drive Files'):
    service = get_drive_service()
    files = list_drive_files(service, FOLDER_ID)
    
    if not files:
        st.write('No files found.')
    else:
        # Create a DataFrame
        df = pd.DataFrame(files)
        df = df[['name', 'webViewLink']]
        df.columns = ['File Name', 'URL']

        # Convert DataFrame to CSV with specified encoding
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        csv_data = csv_buffer.getvalue()

        # Write "Done" message
        st.write("Done")

        # Provide a download button
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name='google_drive_files.csv',
            mime='text/csv'
        )
