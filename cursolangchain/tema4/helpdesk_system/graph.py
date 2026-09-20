from typing import TypedDict, Optional, List, Annotated, Dict, Any
from operator import add

from langchain_google_genai import ChatGoogleGenerativeAI

# Definición del Estado
class HelpDeskState(TypedDict):
    consulta: str
    categoria: str # "automática" o "escalada"
    respuesta_rag: Optional[str]
    confianza: float
    fuentes: List[str]
    contexto_rag: Optional[str]
    requiere_humano: bool
    respuesta_humano: Optional[str]
    respuesta_final: Optional[str]
    historial: Annotated[List[str], add]

class HelpDeskGraph:
    """Grafo del sistema Helpdesk."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="chat-bison@001", temperature=0.2, max_output_tokens=512
            )
        self.state: HelpDeskState = {
            "consulta": "",
            "categoria": "",
            "respuesta_rag": None,
            "confianza": 0.0,
            "fuentes": [],
            "contexto_rag": None,
            "requiere_humano": False,
            "respuesta_humano": None,
            "respuesta_final": None,
            "historial": []
        }

    def update_state(self, updates: Dict[str, Any]):
        for key, value in updates.items():
            if key in self.state:
                self.state[key] = value
                if key != "historial":
                    self.state["historial"].append(f"Updated {key} to {value}")

    def get_state(self) -> HelpDeskState:
        return self.state