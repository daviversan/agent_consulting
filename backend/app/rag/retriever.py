# em backend/app/rag/retriever.py
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.agents import Tool

load_dotenv()
CHROMA_PATH = "storage/vector_db"

def get_retriever():
    """
    Loads the ChromaDB database from disk and returns a retriever object.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    return db.as_retriever(search_kwargs={"k": 10})

def format_docs_with_sources(docs):
    """Formats the retriever's output to include content and source."""
    formatted_docs = []
    for doc in docs:
        source_info = doc.metadata.get("source", "Unknown Source")
        formatted_str = f"Source: {source_info}\nContent: {doc.page_content}"
        formatted_docs.append(formatted_str)
    return "\n\n".join(formatted_docs)

def get_knowledge_base_tool():
    """
    Creates a custom RAG tool that returns content AND source.
    """
    retriever = get_retriever()

    tool = Tool(
        name="consulting_knowledge_base",
        description="Use this tool to search for information and answer specific questions about consulting selection processes, GMAT, business cases, frameworks, and other provided study materials. Do not use it for general summary questions about an entire document.",
        func=lambda query: format_docs_with_sources(retriever.invoke(query))
    )
    return tool