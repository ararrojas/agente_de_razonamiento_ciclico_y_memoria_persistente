"""Esquema del estado del grafo.

El estado nunca se modifica en el lugar: cada nodo devuelve un dict parcial y LangGraph lo
combina con el estado anterior usando el *reducer* declarado en cada campo.
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class EstadoAgente(TypedDict):
    # add_messages concatena (como operator.add) pero además entiende RemoveMessage,
    # que usa el nodo de memoria para sacar mensajes viejos ya resumidos.
    messages: Annotated[list[AnyMessage], add_messages]
    # Resumen acumulado de los turnos que salieron del historial (sin reducer: se reemplaza).
    resumen: str
    # Contador de herramientas pedidas en todo el thread (reducer operator.add: se suma).
    llamadas_herramientas: Annotated[int, operator.add]
    # Cuántas veces se compactó la memoria en este thread.
    compactaciones: Annotated[int, operator.add]
