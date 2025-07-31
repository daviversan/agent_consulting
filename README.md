# Agent Consulting
Acesse a aplicação [aqui](https://agent-consulting-frontend.onrender.com/)

Este projeto consiste em um agente de IA especialista em cases de consultoria. Sua principal função é auxiliar candidatos a se preparar para ‘case interviews’, comuns em processos seletivos para cargos nessas empresas. A primeira versão desse sistema é composta por um Chatbot simples desenvolvido com base no modelo Gemini do Google.  Sua ‘especialização’ em consultoria é alcançada através uma cadeia de prompts utilizando o framework Langchain, que especifica como o bot deve se comportar e sua sequência de ‘pensamentos’. Nesse contexto, o Casebot foi criado para atuar como um “consultor sênior e tutor especialista em processos seletivos de consultoria estratégica”. O `AGENT_SYSTEM_PROMPT` completo pode ser acessado em `backend/app/core/agent.py`
Além de servir como um guia textual, o Casebot gera um código com sintaxe Mermaid, que pode ser colado no editor [Mermaid](https://mermaid.js.org/) para criação de diagramas. Isso é útil para auxiliar o candidato a estruturar o case em frameworks, que são muito utilizados nesse tipo de problema. A cada interação com o bot, o usuário pode se aprofundar em cada bloco do framework e pedir que um novo código seja gerado, para que o diagrama seja atualizado com as novas informações obtidas para a resolução do case.  

## Funcionalidades Principais
- **Agente Especialista:** O bot atua como um consultor sênior, seguindo metodologias de resolução de cases (SCQ, MECE, Árvore de Hipóteses, etc.).
- **Agente Multi-Ferramentas:** O agente é capaz de decidir qual ferramenta usar para a tarefa
- **Calculadora:** Para realizar análises quantitativas durante a resolução de um case.
- **Geração de Frameworks e Diagramas:** O agente estrutura problemas complexos em frameworks e gera o código Mermaid para visualização do fluxo.
- **Memória Conversacional:** O agente mantém o contexto ao longo da conversa para entender perguntas de acompanhamento.
- **[Versões Futuras]:** RAG com Base de Conhecimento Fixa: Responde a perguntas específicas utilizando uma base de conhecimento pré-processada a partir de um Google Drive com materiais de estudo (casebooks, guias de GMAT, etc.).
- **[Versões Futuras]:** RAG Interativo com Upload de Arquivos: Os usuários podem fazer o upload de seus próprios documentos (.pdf) durante uma sessão para que o bot os analise e responda perguntas sobre eles.
- **[Versões Futuras]:** Busca na Base de Conhecimento: Para encontrar fatos e conceitos nos materiais de estudo.

## Arquitetura

O projeto é construído em uma arquitetura de microserviços desacoplada, com um frontend interativo e um backend que orquestra a lógica de IA.

- **Frontend (Streamlit):** Uma interface de chat web onde o usuário interage, faz perguntas e futuramente poderá fazer o upload de arquivos.
- **Backend (FastAPI):** Uma API que recebe as requisições do frontend, gerencia o estado da conversa e orquestra o agente de IA.
- **Orquestração (LangChain):** O Agente, construído com LangChain, atua como o cérebro, interpretando a entrada do usuário, decidindo qual ferramenta usar e formulando a resposta final.
- **LLM (Google Gemini):** Utiliza os modelos da família Gemini (gemini-1.5-flash) para raciocínio, geração de texto e reformulação de perguntas.
- **Vector Stores:**
  - **ChromaDB:** Para o armazenamento persistente da base de conhecimento principal.
  - **FAISS:** Para a criação de índices vetoriais em memória e de alta velocidade para os arquivos enviados pelo usuário.

## Como rodar localmente
Siga estas instruções para configurar e executar o projeto na sua máquina local.

### Pré-requisitos:
- **Git**
- **Python 3.11+**
- **Docker Desktop**
- **Tesseract OCR (Apenas se for implementar RAG com imagens):** Siga as instruções de instalação para o seu sistema operacional. É uma dependência de sistema para o processamento de imagens.

### 1. Configuração do backend

```
# 1. Navegue para a pasta do backend
cd backend

# 2. Crie e ative o ambiente virtual
python -m venv .venv
# No Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# No macOS/Linux:
# source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure os segredos
# Crie um arquivo .env e preencha com suas chaves
cp .env.example .env 
# Crie os arquivos credentials.json e token.json conforme o guia de autenticação do Google

# 5. Faça a ingestão dos dados do Google Drive -> Apenas de for implementar o RAG
# (Execute este passo uma vez para criar o banco de dados vetorial)
python scripts/ingest.py

# 6. Inicie o servidor da API
uvicorn app.main:app --reload
```

### 2. Configuração do Frontend
```
# 1. Abra um NOVO terminal e navegue para a pasta do frontend
cd frontend

# 2. Crie e ative o ambiente virtual
python -m venv .venv
# No Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# No macOS/Linux:
# source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute a aplicação Streamlit
streamlit run app.py
```
## Deploy
Esta aplicação está configurada para deploy na plataforma Render usando Docker, que pode ser acessada pela URL: https://agent-consulting.onrender.com. O processo envolve:

- Um Web Service para o backend, construído a partir do backend/Dockerfile.
- Um Web Service para o frontend, construído a partir do frontend/Dockerfile.

## Próximos passos e melhoria:
- Criar um mecanismo para exportar o relatório final do case para PDF.
- Criar uma feature para uploads de arquivo no chat
- Implementar um sistemad de RAG funcional utilizando ChromaDB para o armazenamento persistente da base de conhecimento principal e FAISS para a criação de índices vetoriais em memória e de alta velocidade para os arquivos enviados pelo usuário.
- Após implementar o sistema de RAG, adicionar um Persistent Disk anexado ao serviço de backend para armazenar o ChromaDB e um Job para executar o script ingest.py e popular o Persistent Disk para o deploy no Render.
