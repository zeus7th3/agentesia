from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
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

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0)
base_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":2})
retriever = MultiQueryRetriever.from_llm(retriever = base_retriever, llm=llm)


consulta = "¿donde se encuentra el local del contrato en el que participa Maria Jimenez Campos"
resultados = retriever.invoke(consulta)

print("top 2 documento mas similares a la consulta : \n ")
for i, doc in enumerate(resultados, start=1):
    print(f"Contenido: {doc.page_content}")
    print(f"metadatos: {doc.metadata}")