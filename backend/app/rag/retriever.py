# Em backend/app/rag/retriever.py

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools.retriever import create_retriever_tool # Importação necessária

load_dotenv()

CHROMA_PATH = "storage/vector_db" 

def get_retriever():
    """
    Carrega o banco de dados ChromaDB do disco e retorna um objeto retriever.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings
    )
    return db.as_retriever(search_kwargs={"k": 5})

def get_retriever_tool():
    """
    Cria uma ferramenta a partir do nosso retriever para ser usada pelo Agente.
    """
    retriever = get_retriever()
    
    # A descrição é a parte mais importante para o agente saber QUANDO usar a ferramenta.
    tool = create_retriever_tool(
        retriever,
        "consulting_knowledge_base",
        "Use esta ferramenta para buscar informações e responder perguntas sobre processos seletivos de consultoria, GMAT, cases de negócio, frameworks e outros materiais de estudo fornecidos. Não a use para perguntas gerais."
    )
    return tool