"""Construcción del StateGraph: memoria -> modelo <-> herramientas."""

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from bibliotecario.catalogo import HOY
from bibliotecario.estado import EstadoAgente
from bibliotecario.memoria import Resumidor, compactar

NODO_MEMORIA = "memoria"
NODO_MODELO = "modelo"
NODO_HERRAMIENTAS = "herramientas"

PROMPT_SISTEMA = f"""Sos "Biblio", el asistente de la Biblioteca Popular del barrio. \
Hoy es {HOY.isoformat()}. Respondé en español rioplatense, breve y amable.

Cómo trabajás:
1. Pensá qué datos necesitás. Si no los tenés, conseguilos con las herramientas: nunca \
inventes libros, socios, fechas ni montos.
2. Encadená herramientas cuando una depende de otra (buscar_socio te da el socio_id que \
piden prestamos_de_socio y calcular_multa).
3. Si una herramienta devuelve "error":
   - si trae una sugerencia que claramente corrige el pedido (por ejemplo un error de \
tipeo en un título), reintentá UNA vez con el valor corregido y aclaralo en tu respuesta;
   - si no podés corregirlo vos, pedile al usuario el dato que falta.
4. Si una herramienta devuelve resultados ambiguos (varias personas o libros posibles), \
no elijas: mostrá las opciones y preguntá.
5. Usá la memoria: si el usuario ya se identificó o dice "ese libro", "yo", "y la multa?", \
resolvelo con lo que ya sabés (historial o resumen) sin volver a preguntar.
6. Con los datos completos, respondé sin llamar más herramientas. Montos en pesos ($1.500)."""

type GrafoBiblioteca = CompiledStateGraph[EstadoAgente, None, EstadoAgente, EstadoAgente]


def armar_prompt(resumen: str) -> SystemMessage:
    if not resumen:
        return SystemMessage(PROMPT_SISTEMA)
    return SystemMessage(
        f"{PROMPT_SISTEMA}\n\nResumen de la conversación anterior con este usuario:\n{resumen}"
    )


def construir_grafo(
    llm: BaseChatModel,
    herramientas: Sequence[BaseTool],
    checkpointer: BaseCheckpointSaver[Any] | None,
    resumidor: Resumidor,
    umbral_resumen: int = 14,
    turnos_a_conservar: int = 2,
) -> GrafoBiblioteca:
    llm_con_herramientas = llm.bind_tools(list(herramientas))

    async def nodo_memoria(state: EstadoAgente) -> dict[str, Any]:
        return await compactar(state, resumidor, umbral_resumen, turnos_a_conservar)

    async def nodo_modelo(state: EstadoAgente) -> dict[str, Any]:
        prompt = armar_prompt(state.get("resumen", ""))
        respuesta = await llm_con_herramientas.ainvoke([prompt, *state["messages"]])
        pedidas = len(respuesta.tool_calls) if isinstance(respuesta, AIMessage) else 0
        return {"messages": [respuesta], "llamadas_herramientas": pedidas}

    grafo = StateGraph(EstadoAgente)
    grafo.add_node(NODO_MEMORIA, nodo_memoria)
    grafo.add_node(NODO_MODELO, nodo_modelo)
    # handle_tool_errors=True: si los argumentos son inválidos o la herramienta lanza una
    # excepción, el error vuelve al modelo como ToolMessage y puede reintentar.
    grafo.add_node(NODO_HERRAMIENTAS, ToolNode(list(herramientas), handle_tool_errors=True))

    grafo.add_edge(START, NODO_MEMORIA)
    grafo.add_edge(NODO_MEMORIA, NODO_MODELO)
    grafo.add_conditional_edges(
        NODO_MODELO, tools_condition, {"tools": NODO_HERRAMIENTAS, END: END}
    )
    grafo.add_edge(NODO_HERRAMIENTAS, NODO_MODELO)
    return grafo.compile(checkpointer=checkpointer)
