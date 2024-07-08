import os
import json
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

SCOPES = ['https://www.googleapis.com/auth/drive']

def get_drive_service(auth_code=None):
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
            flow = InstalledAppFlow.from_client_config(client_secret, SCOPES, redirect_uri=os.getenv("REDIRECT_URI"))
            if auth_code:
                flow.fetch_token(code=auth_code)
                creds = flow.credentials
                with open('token.pickle', 'wb') as token:
                    pickle.dump(creds, token)
            else:
                auth_url, _ = flow.authorization_url(prompt='consent')
                return auth_url
    service = build('drive', 'v3', credentials=creds)
    return service

def list_drive_files(auth_code=None):
    service_or_url = get_drive_service(auth_code)
    if isinstance(service_or_url, str):
        return service_or_url, None
    service = service_or_url
    results = service.files().list(pageSize=10, fields="nextPageToken, files(id, name, webViewLink)").execute()
    items = results.get('files', [])
    return None, items

st.title('Google Drive File Viewer')

if 'auth_code' not in st.session_state:
    st.session_state.auth_code = None

if st.session_state.auth_code:
    url, files = list_drive_files(st.session_state.auth_code)
else:
    url, files = list_drive_files()

if url:
    st.write('Please go to this URL to authorize this application: [Authorization URL]({})'.format(url))
    auth_code = st.text_input('Enter the authorization code:')
    if auth_code:
        st.session_state.auth_code = auth_code
        st.experimental_rerun()
else:
    if not files:
        st.write('No files found.')
    else:
        for file in files:
            st.write(f"{file['name']} - [Open]({file['webViewLink']})")
