import chromadb
import os

# Define o caminho para a pasta do banco de dados
# Lembre-se que este script está em 'scripts/', então usamos '../' para voltar para a raiz do 'backend'
CHROMA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'storage', 'vector_db'))

def main():
    """
    Função principal para se conectar ao DB e inspecionar seu conteúdo.
    """
    if not os.path.exists(CHROMA_PATH):
        print("O diretório do banco de dados não foi encontrado.")
        print(f"Caminho verificado: {CHROMA_PATH}")
        return

    print(f"Conectando ao banco de dados em: {CHROMA_PATH}")

    # 1. Cria um cliente que se conecta ao nosso DB persistente no disco
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # O nome da coleção padrão que o LangChain cria é "langchain"
    # Se você especificou outro nome, mude aqui.
    try:
        collection = client.get_collection(name="langchain")
    except ValueError:
        print("A coleção 'langchain' não foi encontrada.")
        print("Coleções disponíveis:", client.list_collections())
        return

    # 2. Pega o número total de itens (chunks) no banco de dados
    count = collection.count()
    print(f"\nO banco de dados contém um total de {count} itens (chunks).\n")

    # 3. Pega uma amostra dos itens para visualização
    # Vamos pegar os 5 primeiros itens como exemplo.
    # Aumente o 'limit' se quiser ver mais.
    results = collection.get(
        limit=5,
        include=["metadatas", "documents"]  # Pedimos para incluir os metadados e o texto
    )

    print("--- Amostra de 5 Itens do Banco de Dados Vetorial ---\n")

    # 4. Itera sobre os resultados e os exibe de forma legível
    for i, doc_id in enumerate(results["ids"]):
        metadata = results["metadatas"][i]
        document_text = results["documents"][i]

        source = metadata.get("source", "N/A")
        page = metadata.get("page", "N/A")

        print(f"--- Item {i+1} ---")
        print(f"ID: {doc_id}")
        print(f"Fonte do Arquivo: {source}")
        print(f"Página (se aplicável): {page}")
        print(f"Conteúdo (primeiros 200 caracteres):")
        print(f"'{document_text[:200]}...'")
        print("-" * (len("--- Item X ---")))

if __name__ == "__main__":
    main()