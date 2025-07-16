# em backend/scripts/ingest.py
import os
import sys
import io
from dotenv import load_dotenv

# Adiciona o diretório raiz do backend ao path para encontrar o módulo 'app'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importa nossa nova função de autenticação
from app.utils.google_auth import get_drive_service

# Loaders padrão e mais robustos do LangChain
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# --- Configurações ---
CHROMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "vector_db")
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")

def download_and_load_drive_files(service, folder_id):
    """
    Baixa arquivos do Google Drive de forma recursiva e os carrega como Documentos LangChain.
    """
    documents = []
    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    items = results.get("files", [])

    if not items:
        return []

    for item in items:
        # Garante que a variável sempre exista neste loop
        temp_path = None
        file_id = item["id"]
        file_name = item["name"]
        mime_type = item["mimeType"]

        # --- CORREÇÃO 1: CHECA SE O ITEM É UMA PASTA ---
        if mime_type == 'application/vnd.google-apps.folder':
            print(f"Entrando na subpasta: {file_name}...")
            # Chama a função recursivamente para a subpasta
            sub_documents = download_and_load_drive_files(service, file_id)
            documents.extend(sub_documents)
        else:
            # --- SE NÃO FOR UMA PASTA, PROCESSA COMO ARQUIVO ---
            print(f"Processando arquivo: {file_name} ({mime_type})")
            try:
                if mime_type == 'application/vnd.google-apps.document':
                    request = service.files().export_media(fileId=file_id, mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
                    file_ext = '.docx'
                else:
                    request = service.files().get_media(fileId=file_id)
                    file_ext = os.path.splitext(file_name)[1]

                file_content = io.BytesIO(request.execute())

                # --- CORREÇÃO 2: Define temp_path antes do loader ---
                temp_path = f"temp_file{file_ext}"
                with open(temp_path, "wb") as f:
                    f.write(file_content.getbuffer())

                if file_ext.lower() == '.pdf':
                    loader = PyPDFLoader(temp_path)
                elif file_ext.lower() == '.docx':
                    loader = UnstructuredWordDocumentLoader(temp_path)
                else:
                    print(f"Tipo de arquivo não suportado: {mime_type}. Pulando.")
                    os.remove(temp_path)
                    continue

                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file_name
                documents.extend(docs)
                os.remove(temp_path)

            except Exception as e:
                print(f"Erro ao processar o arquivo {file_name}: {e}")
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

    return documents

def ingest_data():
    if not GDRIVE_FOLDER_ID:
        print("Erro: GDRIVE_FOLDER_ID não encontrado no .env.")
        sys.exit(1)

    drive_service = get_drive_service()
    if not drive_service:
        print("Falha na autenticação do Google Drive.")
        sys.exit(1)

    print("Buscando e carregando documentos do Google Drive...")
    documents = download_and_load_drive_files(drive_service, GDRIVE_FOLDER_ID)

    if not documents:
        print("Nenhum documento foi carregado ou processado.")
        return

    print(f"Total de {len(documents)} páginas/documentos carregados.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    splits = text_splitter.split_documents(documents)
    
    if not splits:
        print("A divisão dos documentos resultou em uma lista vazia.")
        return

    print(f"Documentos divididos em {len(splits)} chunks.")
    
    print("Gerando embeddings com a API do Google...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    print(f"Criando e persistindo o banco de dados vetorial em {CHROMA_PATH}...")
    db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print("Ingestão de dados concluída com sucesso! ✅")

if __name__ == "__main__":
    ingest_data()