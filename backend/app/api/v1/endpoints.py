from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Tuple
from app.core.agent import create_agent_executor
from langchain_core.messages import HumanMessage, AIMessage

router = APIRouter()

# Define os modelos de dados para a requisição e resposta da API
class ChatRequest(BaseModel):
    question: str
    chat_history: List[Tuple[str, str]] = Field(
        ...,
        description="Histórico da conversa como uma lista de tuplas (pergunta, resposta)"
    )

class ChatResponse(BaseModel):
    answer: str

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint para receber uma pergunta e retornar a resposta do agente especialista.
    """
    # Cria uma nova instância do executor do agente para cada requisição
    agent_executor = create_agent_executor()
    
    # Converte o histórico de tuplas de texto para o formato de objetos de mensagem
    chat_history_messages = []
    for human_msg, ai_msg in request.chat_history:
        chat_history_messages.append(HumanMessage(content=human_msg))
        chat_history_messages.append(AIMessage(content=ai_msg))

    # Prepara os dados de entrada para o agente
    input_data = {
        "input": request.question,
        "chat_history": chat_history_messages
    }
    
    # Invoca o agente de forma assíncrona para obter a resposta
    # Usamos await para chamadas de rede não bloquearem o servidor
    response = await agent_executor.ainvoke(input_data)

    return ChatResponse(answer=response["output"])