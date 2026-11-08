"""Ejecuta un turno de conversación sobre el grafo, con techo de recursión."""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError

from bibliotecario.grafo import NODO_MODELO, GrafoBiblioteca
from bibliotecario.memoria import texto_de
from bibliotecario.traza import TrazaReAct

log = logging.getLogger(__name__)


def mensaje_de_limite(limite: int) -> str:
    return (
        f"Llegué al máximo de {limite} pasos de razonamiento sin una respuesta segura. "
        "¿Me das un poco más de detalle para intentarlo de nuevo?"
    )


def configuracion(thread_id: str, limite_recursion: int) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": limite_recursion}


async def _cerrar_turno_cortado(grafo: GrafoBiblioteca, config: RunnableConfig, texto: str) -> None:
    """Deja el estado consistente si el turno se cortó por recursion_limit.

    Si el último mensaje guardado pedía herramientas que nunca se ejecutaron, se responden
    con un ToolMessage de cancelación; si no, el próximo turno fallaría en el proveedor.
    """
    estado = await grafo.aget_state(config)
    mensajes = estado.values.get("messages", [])
    respondidas = {m.tool_call_id for m in mensajes if isinstance(m, ToolMessage)}
    pendientes: list[ToolMessage] = []
    for m in mensajes:
        if isinstance(m, AIMessage):
            for llamada in m.tool_calls:
                if llamada["id"] and llamada["id"] not in respondidas:
                    pendientes.append(
                        ToolMessage(
                            content="Cancelada: se alcanzó el límite de pasos.",
                            tool_call_id=llamada["id"],
                            name=llamada["name"],
                            status="error",
                        )
                    )
    await grafo.aupdate_state(
        config, {"messages": [*pendientes, AIMessage(content=texto)]}, as_node=NODO_MODELO
    )


async def ejecutar_turno(
    grafo: GrafoBiblioteca,
    pregunta: str,
    thread_id: str,
    limite_recursion: int,
    turno: int = 1,
    traza: TrazaReAct | None = None,
) -> str:
    config = configuracion(thread_id, limite_recursion)
    if traza:
        traza.pregunta(turno, pregunta)

    entrada: Any = {"messages": [HumanMessage(pregunta)]}
    respuesta = ""
    try:
        async for evento in grafo.astream(entrada, config=config, stream_mode="updates"):
            for nodo, cambios in evento.items():
                if not cambios:
                    continue
                if traza:
                    traza.actualizacion(turno, nodo, cambios)
                for m in cambios.get("messages", []):
                    if isinstance(m, AIMessage) and m.tool_calls:
                        log.info(
                            "thread=%s turno=%d -> %s",
                            thread_id,
                            turno,
                            [c["name"] for c in m.tool_calls],
                        )
                    elif isinstance(m, AIMessage):
                        respuesta = texto_de(m)
    except GraphRecursionError:
        log.warning("thread=%s turno=%d: recursion_limit=%d", thread_id, turno, limite_recursion)
        respuesta = mensaje_de_limite(limite_recursion)
        await _cerrar_turno_cortado(grafo, config, respuesta)
        if traza:
            traza.limite(turno, respuesta)
    return respuesta
