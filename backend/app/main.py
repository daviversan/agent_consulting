from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.v1 import endpoints

app = FastAPI(title="API do Chatbot de Consultoria")

# Configuração do CORS (Cross-Origin Resource Sharing)
# Essencial para permitir que o frontend (ex: http://localhost:8501)
# se comunique com o backend (ex: http://localhost:8000).
origins = [
    "http://localhost",
    "http://localhost:8501", # Porta padrão do Streamlit
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclui as rotas definidas em endpoints.py
app.include_router(endpoints.router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "API do Chatbot de Consultoria está no ar!"}