# Em backend/app/core/agent.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools.render import render_text_description # <-- NOVA IMPORTAÇÃO

# Importa nossas ferramentas
from ..rag.retriever import get_knowledge_base_tool
from .tools import get_calculator_tool

def create_agent_executor():
    """
    Cria o Agente Especialista em Consultoria e seu Executor.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

    # 1. Define a lista de ferramentas que o agente pode usar
    tools = [get_knowledge_base_tool(), get_calculator_tool()]
    
    # 2. Renderiza a descrição das ferramentas em um formato de texto simples
    # Isso torna a existência das ferramentas explícita para o LLM.
    rendered_tools = render_text_description(tools)

    # 3. Cria o novo System Prompt, agora com as ferramentas renderizadas
    AGENT_SYSTEM_PROMPT = f"""
Você é o "CaseBot", um assistente de IA especialista em processos seletivos de consultoria.

**SUA MISSÃO:**
Sua principal função é responder às perguntas do usuário. Antes de responder, você DEVE seguir este processo de raciocínio:
1.  Analise a pergunta do usuário.
2.  Olhe a lista de FERRAMENTAS DISPONÍVEIS abaixo.
3.  Decida se alguma ferramenta é apropriada para a pergunta.
4.  Se uma ferramenta for útil, use-a. Se não, responda usando seu conhecimento geral.

**NÃO** se desculpe por não ter acesso a documentos. Se a pergunta for sobre um documento, use a ferramenta `consulting_knowledge_base`.

**--- FERRAMENTAS DISPONÍVEIS ---**

{rendered_tools}

**--- FLUXO DE RESOLUÇÃO DE CASES ---**
Quando o usuário pedir para resolver um case, siga o fluxo de forma interativa, parando a cada passo para receber o input do usuário (Definição do Problema, Estrutura, Análise de Hipóteses, etc.).

Responda sempre em português brasileiro.
"""

    # 4. Cria o prompt do agente com as novas instruções
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", AGENT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    
    # 5. Cria o agente e o executor (continua igual)
    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, # Mantenha verbose=True para depuração!
    )

    return agent_executor