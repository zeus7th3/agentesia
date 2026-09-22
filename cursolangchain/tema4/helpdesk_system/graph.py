from typing import TypedDict, Optional, List, Annotated, Dict, Any
from operator import add

from langchain_google_genai import ChatGoogleGenerativeAI
from rag_system import VectorRAGSystem
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from config import CHROMADB_PATH, GENERATIVE_MODEL

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
        self.llm = ChatGoogleGenerativeAI(model=GENERATIVE_MODEL, temperature=0.1)
        self.rag = VectorRAGSystem(chroma_path=CHROMADB_PATH)
        self.graph = None

    def procesar_rag(self, state):
        """"Buscar el contexto de la consutal usando el sitema RAG"""
        consulta = state["consulta"]
        resultado = self.rag.buscar(consulta)
        return {
            "respuesta_rag" : resultado["respuesta"],
            "confianza" : resultado["confianza"],
            "fuentes" : resultado["fuentes"],
            "contexto_rag" : resultado["respuesta"],
            "historial" : [
                f"RAG ejecutando con MultiQueryRetriever",
                f"Confianza: {resultado['confianza']}", 
                f"Fuentes consultadas: {len(resultado['fuentes'])}"
            ]
        }

    def clasificar_con_contexto(self, state):
        """Clasificar la consulta para responder automáticamente o escalar usando el contexto de RAG"""
        consulta = state['consulta']
        contexto_rag = state.get('contexto_rag','')
        confianza = state.get('confianza', 0.0)

        prompt = ChatPromptTemplate.from_template(
            """Eres un experto asistente de helpdesk. 
            Analiza esta consulta de helpdesk y decide si puede responderse automaticamente o necesita ser escalado
            CONSULTA DEL USUARIO: {consulta}

            INFORMACION ENCONTRADA EN LA BSAE DE CONOCIMIENTO: {contexto_rag}

            CONFIANZA DE LA BUSQUEDA: {confianza}

            Criterios de decisión:
            - AUTOMATICA: Si la información de la BD responde completamente la consulta,
            tiene buena confianza (>0.6), y es un tema estándar/procedimiento conocido.
            - ESCALADA: Si la información es insufieciente, confianza baja, problema complejo/único,
            requiere acceso a sistemas internos o involucrea decisiones de negocio.

            Responde solo con la categoría: "automática" o "escalada" y una breve justificación (máximo 20 palabras):
            """            
        )

        try:
            response = self.llm.invoke(prompt.format(
                consulta = consulta,
                contexto_rag = contexto_rag,
                confianza = confianza
            ))

            content = response.content.strip().lower()
            if "automática" in content:
                categoria = "automática"
                justificacion = content.split("automática")[-1].strip()
            elif "escalada" in content:
                categoria = "escalada"
                justificacion = content.split("escalada")[-1].strip()
            else:
                categoria = "escalada"
                justificacion = "No se pudo determinar la categoría, se escalará por seguridad."

            return {
                "categoria": categoria,                
                "historial": [
                    f"Clasificación con contexto: {categoria}",
                    f"Justificación: {justificacion}"
                ]
            }
                
        except Exception as e:
            categoria = "automática" if confianza > 0.6 else "escalada"
            return {
                "categoria": categoria,
                "historial": [
                    f"Error al clasificar: {str(e)}",
                    f"Se asigna categoría por confianza: {categoria}"
                ]
            }

    def preparar_escalado(self, state):
        """Preparar la información para el escalado a un humano"""
        return {
            "requiere_humano": True,
            "historial": [
                f"Escalado a agente humano - Esperando intervención."
            ]
        }

    def procesar_respuesta_humana(self, state):
        """Procesar la respuesta del humano"""
        respuesta_huamana = state.get("respuesta_humano", "")
        if respuesta_huamana:
            return {
                "respuesta_final": respuesta_huamana,
                "historial": [
                    f"Agente humano proporcionó respuesta."
                ]
            }
        return {
            "historial": [
                f"Esperando respuesta del agente humano."
            ]
        }

    def generar_respuesta_final(self, state):
        """Generar la respuesta final al usuario, ya sea automática o del humano"""
        categoria = state.get("categoria")

        if categoria == "automática":
            respuesta_final = state.get("respuesta_rag", "")
            fuentes = state.get("fuentes", [])

            # Enriquecer respuesta final con fuentes si existen
            if fuentes:
                fuentes_texto = ", ".join(fuentes)
                respuesta_final += f"\n\nFuentes consultadas: {fuentes_texto}"
            return {
                "respuesta_final": respuesta_final,
                "historial": [
                    f"Respuesta generada automáticamente."
                ]
            }
        elif categoria == "escalada":
            respuesta_final = state.get("respuesta_humano", "")
            return {
                "respuesta_final": respuesta_final,
                "historial": [
                    f"Respuesta proporcionada por agente humano."
                ]
            }
        else:
            return {
                "respuesta_final": None,
                "historial": [
                    f"No se pudo generar respuesta final."
                ]
            }

    # Funciones de enrutamiento
    def decidir_desde_clasificación(self, state):
        """Decidir el siguiente paso basado en la clasificación"""
        categoria = state.get("categoria","escalada")  # Por defecto, escalar si no se puede clasificar
        return categoria
        
        #if categoria == "automática":
        #    return "generar_respuesta_final"
        #elif categoria == "preparar_escalado":        
        #    return self.preparar_escalado(state)
        #else:
        #    return {
        #        "historial": [
        #            f"Categoría desconocida: {categoria}. No se puede decidir el siguiente paso."
        #        ]
        #    }

    def decidir_desde_humano(self, state):
        """Decidir si continuar o esperar respuesta humana """
        respuesta_humana = state.get("respuesta_humano")
        if respuesta_humana:
            return "procesar_respuesta_humana"
        else:
            return "esperar"

    def crear_grafo(self):
        """Crear el grafo de Langraph con los nodos y control de flujo"""
        graph = StateGraph(HelpDeskState)

        # Agregar nodos al grafo
        graph.add_node("procesar_rag", self.procesar_rag)
        graph.add_node("clasificar_con_contexto", self.clasificar_con_contexto)
        graph.add_node("preparar_escalado", self.preparar_escalado)
        graph.add_node("procesar_respuesta_humana", self.procesar_respuesta_humana)
        graph.add_node("generar_respuesta_final", self.generar_respuesta_final)

        # Definir la estructura del grafo y las transiciones        
        graph.add_edge(START, "procesar_rag")
        graph.add_edge("procesar_rag", "clasificar_con_contexto")

        #Edges condicionales del grafo
        graph.add_conditional_edges(
            "clasificar_con_contexto",
            self.decidir_desde_clasificación,
            {
                "automática": "generar_respuesta_final",
                "escalada": "preparar_escalado"
            }
        )
        graph.add_conditional_edges(
            "preparar_escalado",
            self.decidir_desde_humano,
            {
                "procesar_respuesta_humana": "procesar_respuesta_humana",
                "esperar": END # Pausar la ejecución del grafo haste que responda el humano
            }
        )

        graph.add_edge("procesar_respuesta_humana", END)
        graph.add_edge("generar_respuesta_final", END)

        self.graph = graph

        return graph

    def compilar(self):
        """Compila el grafo con checkpointer."""
        if not self.graph:
            self.crear_grafo()

        conn = sqlite3.connect("helpdesk.db",check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        compiled = self.graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["procesar_respuesta_humana"]
            )

        
        return compiled

def crear_helpdesk():
    helpdesk = HelpDeskGraph()
    return helpdesk.compilar()