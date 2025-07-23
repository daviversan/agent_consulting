# Em backend/app/rag/retriever.py

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import Tool


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
    return db.as_retriever(search_kwargs={"k": 3})


def get_knowledge_base_tool():
    """
    Cria uma ferramenta de RAG customizada que retorna o conteúdo e a fonte.
    """
    retriever = get_retriever()
    
    tool = Tool(
        name="consulting_knowledge_base",
        description="Use esta ferramenta para buscar informações e responder perguntas sobre processos seletivos de consultoria, GMAT, cases de negócio, frameworks e outros materiais de estudo fornecidos. Não a use para perguntas gerais.",
        # A função da ferramenta agora formata a saída para incluir a fonte
        func=lambda query: format_docs_with_sources(retriever.invoke(query))
    )
    return tool

def format_docs_with_sources(docs):
    """Formata a saída do retriever para incluir o conteúdo e a fonte."""
    formatted_docs = []
    for doc in docs:
        # Cria uma string formatada para cada documento
        source_info = doc.metadata.get("source", "Fonte desconhecida")
        formatted_str = f"Fonte: {source_info}\nConteúdo: {doc.page_content}"
        formatted_docs.append(formatted_str)
    # Junta todos os documentos formatados em uma única string
    return "\n\n".join(formatted_docs)