import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("--- MODELOS DE EMBEDDING REALES PARA TU API KEY ---")
try:
    modelos_encontrados = 0
    for m in genai.list_models():
        # Filtramos estrictamente por capacidad de crear vectores (embeddings)
        if 'embedContent' in m.supported_generation_methods:
            print(f"Nombre exacto a usar: {m.name}")
            modelos_encontrados += 1
            
    if modelos_encontrados == 0:
        print("⚠️ Tu API Key no tiene acceso a NINGÚN modelo de embeddings de Google.")
except Exception as e:
    print(f"Error de conexión: {e}")