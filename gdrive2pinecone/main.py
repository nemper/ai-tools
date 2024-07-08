import os
import json
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
import pandas as pd
from io import StringIO

SCOPES = ['https://www.googleapis.com/auth/drive']

def get_drive_service():
    client_secret_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    service_account_info = json.loads(client_secret_json)
    credentials = service_account.Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    service = build('drive', 'v3', credentials=credentials)
    return service

def list_drive_files():
    service = get_drive_service()
    results = service.files().list(pageSize=100, fields="nextPageToken, files(id, name, webViewLink)").execute()
    items = results.get('files', [])
    return items

st.title('Google Drive File Viewer')

if st.button('List Google Drive Files'):
    files = list_drive_files()
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