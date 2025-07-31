import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def get_drive_service():
    creds = None
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(os.path.dirname(script_dir))
    token_path = os.path.join(backend_dir, "token.json")
    credentials_path = os.path.join(backend_dir, "credentials.json")

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path, 
                SCOPES,
                redirect_uri='urn:ietf:wg:oauth:2.0:oob' 
            )
            # Gera a URL de autorização
            auth_url, _ = flow.authorization_url(prompt='consent')
            
            print('Please go to this URL and authorize access:')
            print(auth_url)
            
            # Pede para o usuário colar o código
            code = input('Enter the authorization code here: ')
            flow.fetch_token(code=code)
            creds = flow.credentials

        with open(token_path, "w") as token:
            token.write(creds.to_json())
    
    try:
        service = build("drive", "v3", credentials=creds)
        print("Serviço do Google Drive autenticado com sucesso.")
        return service
    except HttpError as error:
        print(f"Ocorreu um erro ao criar o serviço do Drive: {error}")
        return None