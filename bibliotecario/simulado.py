"""Modelo guionado: reemplaza al LLM para correr la demo y los tests sin API key.

Las herramientas, el grafo, el checkpointer y la memoria son los reales; solo las
decisiones del modelo vienen de un guion fijo.
"""

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field

PREGUNTAS_DEMO: list[str] = [
    "¿Tienen disponible Rayuela?",
    "Soy Ana Torres. ¿Qué libros tengo prestados?",
    "¿Y tengo que pagar alguna multa?",
    "¿Hay algún ejemplar de Cien años de solead?",
    "Quiero ver los préstamos de Pérez.",
    "Camila.",
]

PREGUNTA_TRAS_REINICIO = "Antes de irme: ¿cuánto era la multa que me salió a mí?"


def accion(nombre: str, args: dict[str, Any], id_: str, pensamiento: str = "") -> AIMessage:
    return AIMessage(
        content=pensamiento,
        tool_calls=[{"name": nombre, "args": args, "id": id_, "type": "tool_call"}],
    )


def guion_demo() -> list[AIMessage]:
    return [
        # Turno 1: una herramienta.
        accion(
            "buscar_libro",
            {"consulta": "Rayuela"},
            "c1",
            "Necesito ver el catálogo para saber si quedan ejemplares.",
        ),
        AIMessage(
            content=(
                "Rayuela (Julio Cortázar, L-01) no tiene ejemplares libres ahora: los 2 están "
                "prestados. La devolución más próxima vencía el 15/10/2026, así que podría "
                "volver en cualquier momento."
            )
        ),
        # Turno 2: dos herramientas encadenadas (nombre -> socio_id -> préstamos).
        accion(
            "buscar_socio",
            {"nombre": "Ana Torres"},
            "c2",
            "Primero necesito el socio_id de Ana Torres.",
        ),
        accion(
            "prestamos_de_socio",
            {"socio_id": 11},
            "c3",
            "Ana es la socia 11; ahora consulto sus préstamos.",
        ),
        AIMessage(
            content=(
                "Ana, tenés 2 libros: Rayuela (venció el 15/10/2026, 22 días de atraso) y "
                "El Aleph (vence el 08/11/2026, al día)."
            )
        ),
        # Turno 3: usa la memoria (socio 11) sin volver a preguntar.
        accion(
            "calcular_multa", {"socio_id": 11}, "c4", "Ya sé que es la socia 11; calculo la multa."
        ),
        AIMessage(
            content=(
                "Sí: tenés una multa de $3.300 por los 22 días de atraso de Rayuela. Hasta "
                "pagarla no podés pedir libros nuevos."
            )
        ),
        # Turno 4: error de la herramienta -> segundo intento con la sugerencia.
        accion("buscar_libro", {"consulta": "Cien años de solead"}, "c5"),
        accion(
            "buscar_libro",
            {"consulta": "Cien años de soledad"},
            "c6",
            "No hubo resultados, pero la herramienta sugiere 'Cien años de soledad': "
            "es claramente un error de tipeo, reintento.",
        ),
        AIMessage(
            content=(
                "Supuse que buscabas *Cien años de soledad* (García Márquez): hay 2 ejemplares "
                "disponibles de 3."
            )
        ),
        # Turno 5: resultado ambiguo -> pide aclaración.
        accion("buscar_socio", {"nombre": "Pérez"}, "c7"),
        AIMessage(
            content=(
                "Hay dos socios con ese apellido: Bruno Pérez (socio 12) y Camila Pérez "
                "(socio 13). ¿De cuál querés ver los préstamos?"
            )
        ),
        # Turno 6: la aclaración se resuelve con el turno anterior.
        accion(
            "prestamos_de_socio",
            {"socio_id": 13},
            "c8",
            "Camila Pérez es la socia 13 (lo vi en el turno anterior).",
        ),
        AIMessage(
            content=(
                "Camila Pérez tiene 1 libro: La casa de los espíritus, vence el 15/11/2026 "
                "(al día)."
            )
        ),
        # Turno 7 (después de reiniciar el proceso): responde desde la memoria persistida.
        AIMessage(
            content=(
                "Tu multa, Ana, era de $3.300 por el atraso de Rayuela (lo tengo del resumen "
                "de nuestra charla)."
            )
        ),
    ]


class ModeloGuionado(BaseChatModel):
    """Devuelve las respuestas del guion en orden y guarda lo que recibió en cada llamada."""

    respuestas: list[AIMessage]
    posicion: int = 0
    recibido: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "guionado"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.posicion >= len(self.respuestas):
            raise RuntimeError("Se terminó el guion del modelo simulado.")
        self.recibido.append(list(messages))
        respuesta = self.respuestas[self.posicion].model_copy()
        self.posicion += 1
        return ChatResult(generations=[ChatGeneration(message=respuesta)])
