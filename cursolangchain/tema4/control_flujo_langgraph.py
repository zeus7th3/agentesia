from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    numero: int
    resultado: str

graph = StateGraph(State)

def caso_par(state):
    return{'resultado': 'El número es par'}

def caso_impar(state):
    return{'resultado': 'El número es impar'}

graph.add_node("Par", caso_par)
graph.add_node("Impar", caso_impar)

# Definir la funcion de rountin para dcidir la rama de ejecución
def decidir_rama(state):
    if state["numero"]%2==0:
        return "Par"
    else:
        return "Impar"
    
# Añadir el edbe condicional al workflow
graph.add_conditional_edges(START, decidir_rama)

# Conectar ambos casos al final
graph.add_edge("Par",END)
graph.add_edge("Impar",END)

compiled = graph.compile()

# Probar el grafo
print (compiled.invoke({"numero":4.4})["resultado"])


