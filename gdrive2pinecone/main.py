import os
import json
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/drive']

def get_drive_service():
    client_secret_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    service_account_info = json.loads(client_secret_json)
    credentials = service_account.Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    service = build('drive', 'v3', credentials=credentials)
    return service

def list_drive_files():
    service = get_drive_service()
    results = service.files().list(pageSize=10, fields="nextPageToken, files(id, name, webViewLink)").execute()
    items = results.get('files', [])
    return items

st.title('Google Drive File Viewer')

if st.button('List Google Drive Files'):
    files = list_drive_files()
    if not files:
        st.write('No files found.')
    else:
        for file in files:
            st.write(f"{file['name']} - [Open]({file['webViewLink']})")
