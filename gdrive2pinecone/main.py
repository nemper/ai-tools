import os
import json
import google.auth
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from flask import Flask, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = 'YOUR_SECRET_KEY'  # Replace with your secret key

# Path to the credentials file
CREDENTIALS_FILE = 'credentials.json'
SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']

@app.route('/')
def index():
    return 'Google Drive File URL Extractor'

@app.route('/authorize')
def authorize():
    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
    authorization_url, state = flow.authorization_url(access_type='offline', include_granted_scopes='true')
    session['state'] = state
    return redirect(authorization_url)

@app.route('/oauth2callback')
def oauth2callback():
    state = session['state']
    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES, state=state)
    flow.fetch_token(authorization_response=request.url)

    credentials = flow.credentials
    session['credentials'] = credentials_to_dict(credentials)

    return redirect(url_for('list_files'))

@app.route('/list_files')
def list_files():
    if 'credentials' not in session:
        return redirect(url_for('authorize'))

    credentials = Credentials(**session['credentials'])
    service = build('drive', 'v3', credentials=credentials)

    results = service.files().list(pageSize=10, fields="files(id, name, webViewLink)").execute()
    items = results.get('files', [])

    if not items:
        return 'No files found.'
    
    file_urls = {item['name']: item['webViewLink'] for item in items}

    return json.dumps(file_urls, indent=2)

def credentials_to_dict(credentials):
    return {'token': credentials.token, 'refresh_token': credentials.refresh_token, 'token_uri': credentials.token_uri, 'client_id': credentials.client_id, 'client_secret': credentials.client_secret, 'scopes': credentials.scopes}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)