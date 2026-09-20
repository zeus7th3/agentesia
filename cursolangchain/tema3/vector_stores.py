from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

loader = PyPDFDirectoryLoader("/home/tomcat/proyectos/udemy/agentesia/cursolangchain/tema3/contratos")
documentos = loader.load()

print(f"Se cargaron {len(documentos)} documentos desde el escritorio.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=1000
)

docs_split = text_splitter.split_documents(documentos)

print(f"Se crearon {len(docs_split)} chunks de texto.")

vectorstore = Chroma.from_documents(
    docs_split,
    embedding= GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key = api_key
        ),
    persist_directory="/home/tomcat/proyectos/udemy/agentesia/cursolangchain/tema3/chroma_db"
)

consulta = "¿donde se encuentra el local del contrato en el que participa Maria Jimenez Campos"
resultados = vectorstore.similarity_search(consulta, k=3)

print("top 3 documento mas similares a la consulta : \n")
for i, doc in enumerate(resultados, start=1):
    print(f"Contenido: {doc.page_content}")
    print(f"metadatos: {doc.metadata}")