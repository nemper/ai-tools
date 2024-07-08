import os
import json
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

SCOPES = ['https://www.googleapis.com/auth/drive']

def get_drive_service():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            client_secret_json = os.getenv("GOOGLE_CLIENT_SECRET_JSON")
            client_secret = json.loads(client_secret_json)
            flow = InstalledAppFlow.from_client_config(client_secret, SCOPES)
            creds = flow.run_local_server(port=0)
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
    service = build('drive', 'v3', credentials=creds)
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

if __name__ == '__main__':
    st.run()