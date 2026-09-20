# Configuración de modelos
EMBEDDING_MODEL = "models/gemini-embedding-2"
QUERY_MODEL = "models/gemini-2.5-flash-lite"
GENERARION_MODEL = "models/gemini-2.5-flash-lite"

# Configuración del vector store
CHROMA_DB_PATH = "/home/tomcat/proyectos/udemy/agentesia/cursolangchain/tema3/chroma_db"

# Configuración del retriever
SEARCH_TYPE = "mmr"
SEARCH_K = 2
MMR_DIVERSITY_LAMBDA = 0.7
MMR_FETCH_K = 20

# Configuración alternativa para retriever híbrido
ENABLE_HYBRID_SEARCH = True
SIMILARITY_THRESHOLD = 0.70