"""Gestión del contexto: resume los turnos viejos para que el estado no crezca sin límite."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)

from bibliotecario.estado import EstadoAgente

type Resumidor = Callable[[str, Sequence[BaseMessage]], Awaitable[str]]

INSTRUCCION_RESUMEN = """Resumí en español, en no más de 6 viñetas, la conversación entre un \
usuario y el asistente de una biblioteca. Conservá SIEMPRE los datos concretos que puedan \
servir después: nombres, socio_id, libro_id, títulos, fechas, montos y pedidos pendientes \
de aclaración. No agregues nada que no esté en la conversación."""


def texto_de(mensaje: BaseMessage) -> str:
    """Texto plano de un mensaje (el contenido puede venir en bloques)."""
    contenido = mensaje.content
    if isinstance(contenido, str):
        return contenido
    partes: list[str] = []
    for bloque in contenido:
        if isinstance(bloque, str):
            partes.append(bloque)
        elif bloque.get("type") == "text":
            partes.append(str(bloque.get("text", "")))
    return "".join(partes)


def _transcribir(mensajes: Sequence[BaseMessage]) -> str:
    lineas: list[str] = []
    for m in mensajes:
        if isinstance(m, HumanMessage):
            lineas.append(f"Usuario: {texto_de(m)}")
        elif isinstance(m, ToolMessage):
            lineas.append(f"Resultado de {m.name}: {texto_de(m)[:400]}")
        elif isinstance(m, AIMessage) and m.tool_calls:
            for llamada in m.tool_calls:
                lineas.append(f"Asistente llama a {llamada['name']}({llamada['args']})")
        elif isinstance(m, AIMessage):
            lineas.append(f"Asistente: {texto_de(m)}")
    return "\n".join(lineas)


def resumidor_con_llm(llm: BaseChatModel) -> Resumidor:
    """Resumidor que le pide al LLM (sin herramientas) un resumen acumulativo."""

    async def resumir(resumen_previo: str, mensajes: Sequence[BaseMessage]) -> str:
        previo = f"Resumen anterior:\n{resumen_previo}\n\n" if resumen_previo else ""
        pedido = f"{previo}Conversación nueva a incorporar:\n{_transcribir(mensajes)}"
        respuesta = await llm.ainvoke([SystemMessage(INSTRUCCION_RESUMEN), HumanMessage(pedido)])
        return texto_de(respuesta).strip()

    return resumir


async def resumidor_extractivo(resumen_previo: str, mensajes: Sequence[BaseMessage]) -> str:
    """Resumidor determinístico (sin LLM): conserva pregunta y respuesta final de cada turno.

    Se usa en el modo offline y en los tests.
    """
    lineas = [resumen_previo] if resumen_previo else []
    for m in mensajes:
        if isinstance(m, HumanMessage):
            lineas.append(f"- Usuario preguntó: {texto_de(m)}")
        elif isinstance(m, AIMessage) and not m.tool_calls:
            lineas.append(f"  Asistente respondió: {texto_de(m)}")
    return "\n".join(lineas)


def punto_de_corte(mensajes: Sequence[BaseMessage], turnos_a_conservar: int) -> int:
    """Índice desde el cual se conservan los mensajes (0 = no se corta nada).

    Siempre corta justo antes de un HumanMessage, así nunca separa una llamada a
    herramienta de su resultado (los proveedores rechazan tool_calls sin respuesta).
    """
    humanos = [i for i, m in enumerate(mensajes) if isinstance(m, HumanMessage)]
    if len(humanos) <= turnos_a_conservar:
        return 0
    return humanos[-turnos_a_conservar]


async def compactar(
    state: EstadoAgente, resumidor: Resumidor, umbral: int, turnos_a_conservar: int
) -> dict[str, Any]:
    """Si el historial supera el umbral, resume los turnos viejos y los quita del estado.

    Devuelve una actualización parcial: RemoveMessage por cada mensaje resumido (lo
    interpreta el reducer add_messages), el resumen nuevo y +1 en el contador.
    """
    mensajes = state["messages"]
    if len(mensajes) <= umbral:
        return {}
    corte = punto_de_corte(mensajes, turnos_a_conservar)
    if corte == 0:
        return {}
    viejos = mensajes[:corte]
    resumen = await resumidor(state.get("resumen", ""), viejos)
    return {
        "resumen": resumen,
        "messages": [RemoveMessage(id=m.id) for m in viejos if m.id],
        "compactaciones": 1,
    }
