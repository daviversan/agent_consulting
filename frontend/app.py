import streamlit as st
import requests


# URL do endpoint da nossa API FastAPI
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/api/v1/chat")

# Título da Aplicação
st.title("🤖 CaseBot - Seu Tutor de Consultoria")
st.caption("Um chatbot para te ajudar a se preparar para entrevistas de consultoria.")

# Inicializa o histórico de conversas no estado da sessão do Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! Como posso te ajudar a se preparar hoje?"}
    ]

# Exibe as mensagens do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Captura a nova pergunta do usuário
if prompt := st.chat_input("Faça sua pergunta..."):
    # Adiciona a mensagem do usuário ao histórico e exibe na tela
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- CORREÇÃO 1: Preparação do Histórico de Conversa ---
    # Transforma o histórico da sessão no formato de tuplas (pergunta, resposta) que a API espera.
    # Esta abordagem é mais robusta e evita o bug da versão anterior.
    chat_history_tuples = []
    # Itera sobre as mensagens, pulando a primeira mensagem de boas-vindas do assistente
    for i in range(1, len(st.session_state.messages) -1, 2):
        user_message = st.session_state.messages[i]
        assistant_message = st.session_state.messages[i+1]
        if user_message["role"] == "user" and assistant_message["role"] == "assistant":
            chat_history_tuples.append(
                (user_message["content"], assistant_message["content"])
            )

    api_payload = {
        "question": prompt,
        "chat_history": chat_history_tuples
    }

    # Mostra um "spinner" enquanto espera a resposta da API
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                # --- MELHORIA 2: Envio de Dados para a API ---
                # Usar o parâmetro 'json' é a forma recomendada pela biblioteca requests.
                # Ele automaticamente formata os dados e define o cabeçalho correto.
                response = requests.post(BACKEND_URL, json=api_payload)
                response.raise_for_status() # Lança um erro para respostas ruins (4xx ou 5xx)
                
                # Extrai a resposta da API
                answer = response.json().get("answer")
                st.markdown(answer)
                
                # Adiciona a resposta do assistente ao histórico
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except requests.exceptions.RequestException as e:
                st.error(f"Erro ao conectar com o backend: {e}")