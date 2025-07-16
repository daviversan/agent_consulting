# Em backend/app/core/tools.py

from langchain.chains import LLMMathChain
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI

def get_calculator_tool():
    """
    Cria uma ferramenta de calculadora que usa um LLM para resolver problemas matemáticos.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)

    # LLMMathChain é uma cadeia especializada em interpretar e resolver problemas matemáticos
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

    tool = Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="Use esta ferramenta para resolver qualquer questão matemática quantitativa. O input deve ser uma pergunta matemática completa."
    )
    return tool