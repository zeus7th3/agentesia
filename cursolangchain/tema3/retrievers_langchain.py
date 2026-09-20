from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

vectorstore = Chroma(
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key = api_key
        ),
    persist_directory="/home/tomcat/proyectos/udemy/agentesia/cursolangchain/tema3/chroma_db"
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":2})

consulta = "¿donde se encuentra el local del contrato en el que participa Maria Jimenez Campos"
resultados = retriever.invoke(consulta)

print("top 3 documento mas similares a la consulta : \n")
for i, doc in enumerate(resultados, start=1):
    print(f"Contenido: {doc.page_content}")
    print(f"metadatos: {doc.metadata}")