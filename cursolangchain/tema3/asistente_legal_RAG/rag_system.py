from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import EnsembleRetriever
import streamlit as st
from dotenv import load_dotenv
from config import *
from prompts import *
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

@st.cache_resource
def initialize_rag_system():
    # Vector strore
    vectorstore = Chroma(
        embedding_function=GoogleGenerativeAIEmbeddings (
            model=EMBEDDING_MODEL,
            google_api_key = api_key),
        persist_directory=CHROMA_DB_PATH
    )

    # Modelos
    llm_queries = ChatGoogleGenerativeAI(
        model = QUERY_MODEL,
        temperature=0,
        google_api_key = api_key)
    llm_generation = ChatGoogleGenerativeAI(
        model=GENERARION_MODEL,
        temperature=0,
        google_api_key = api_key)

    # Retriever MM (mazimal margin relevance)
    base_retriever = vectorstore.as_retriever(
        search_type = SEARCH_TYPE,
        search_kwargs={
            "k": SEARCH_K,
            "lambda" : MMR_DIVERSITY_LAMBDA,
            "fetch_k" : MMR_FETCH_K
        }        
    )

    # Retriever adicional con similarty para comparar
    similarity_retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k":SEARCH_K}
    )

    # Prompt personalizado para MultiQueryRetriever
    multi_query_prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT)

    # MultiqueryRetriever con prompt personalizado
    mmr_multi_retriever = MultiQueryRetriever.from_llm(
        retriever = base_retriever,
        llm=llm_queries,
        prompt = multi_query_prompt        
    )

    # Ensemble Retriever que combinar MMR y SIMILARITY
    if ENABLE_HYBRID_SEARCH:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[mmr_multi_retriever, similarity_retriever],
            weights = [0.7,0.3], # Mayor peso a MMR
            similarity_threshold = SIMILARITY_THRESHOLD
        ) 
        final_retriever = ensemble_retriever
    
    else:
        final_retriever = mmr_multi_retriever


    prompt = PromptTemplate.from_template(RAG_TEMPLATE)

    # Funcion para formatera y preprocesar 
    def format_docs(docs):
        formmated = []

        for i, doc in enumerate(docs, 1):
            header = f"[Fragmento {i}]"
            if doc.metadata:
                if 'source' in doc.metadata:
                    source = os.path.basename(doc.metadata['source'])
                    #source = doc.metadata['source'].split("\\")[-1] if '\\' in doc.metadata['source'] else doc.metadata['source']
                    header += f" - Fuente: {source}"
                if 'page' in doc.metadata:
                    header += f" - Página: {doc.metadata['page']}"

            content = doc.page_content.strip()
            formmated.append(f"{header}\n{content}")
        return "\n\n".join(formmated)

    rag_chain = (
        {
            "context" : final_retriever | format_docs,
            "question" : RunnablePassthrough()
        }
        
        | prompt
        | llm_generation 
        | StrOutputParser()
    ) 

    return rag_chain, mmr_multi_retriever

def query_rag(question):
    try:
        rag_chain, retriever = initialize_rag_system()

        # Obtener respuesta
        response = rag_chain.invoke(question)

        # Obtener documentos para mostralos
        docs = retriever.invoke(question)

        # Formatear los documentos para 
        docs_info_list =[]
        for i, doc in enumerate(docs[:SEARCH_K],1):
            source_raw = doc.metadata.get('source','No especificada')
            docs_info = {
                "fragmento" : i,
                "contenido": doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content,
                "fuente": os.path.basename(source_raw) if source_raw != "No especificado" else source_raw,
                #"fuente": doc.metadata.get('source', 'No especificada'.split("\\")[:-1]),
                "pagina": doc.metadata.get('page','No especificada')
            }
        docs_info_list.append(docs_info)

        return response, docs_info_list
    
    except Exception as e:
        error_msg = f"Error al procesar la consulta: {str(e)}"
        return error_msg, []
    
def get_retriever_info():
    """Obtiene información sobre la configuración del retriever"""
    return{
        "tipo": f"{SEARCH_TYPE.upper()} + MultiQuery" + (" +  Hybrid" if ENABLE_HYBRID_SEARCH else ""),
        "documentos" : SEARCH_K,
        "diversidad" : MMR_DIVERSITY_LAMBDA,
        "candidatos" : MMR_FETCH_K,
        "umbral" : SIMILARITY_THRESHOLD if ENABLE_HYBRID_SEARCH else "N/A"
    }

