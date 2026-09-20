import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configura la API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("--- MODELOS DISPONIBLES PARA TU API KEY ---")
try:
    for m in genai.list_models():
        # Filtramos solo los que sirven para generar texto (chat)
        if 'generateContent' in m.supported_generation_methods:
            print(f"Nombre: {m.name}")
except Exception as e:
    print(f"Error de conexión: {e}")