"""Registro de la traza ReAct (Pensamiento -> Acción -> Observación -> Respuesta)."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from bibliotecario.memoria import texto_de


class TipoPaso(StrEnum):
    PREGUNTA = "pregunta"
    PENSAMIENTO = "pensamiento"
    ACCION = "accion"
    OBSERVACION = "observacion"
    MEMORIA = "memoria_compactada"
    RESPUESTA = "respuesta"
    LIMITE = "limite_recursion"


@dataclass(slots=True)
class Paso:
    n: int
    turno: int
    tipo: TipoPaso
    nodo: str
    hora: str = field(default_factory=lambda: datetime.now().isoformat(timespec="milliseconds"))
    texto: str | None = None
    herramienta: str | None = None
    argumentos: dict[str, Any] | None = None
    resultado: Any = None
    es_error: bool = False


def _a_json(texto: str) -> Any:
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        return texto


class TrazaReAct:
    def __init__(self, thread_id: str, metadatos: dict[str, Any] | None = None) -> None:
        self.thread_id = thread_id
        self.metadatos: dict[str, Any] = metadatos or {}
        self.pasos: list[Paso] = []

    def _nuevo(self, turno: int, tipo: TipoPaso, nodo: str, **campos: Any) -> None:
        self.pasos.append(Paso(n=len(self.pasos) + 1, turno=turno, tipo=tipo, nodo=nodo, **campos))

    def pregunta(self, turno: int, texto: str) -> None:
        self._nuevo(turno, TipoPaso.PREGUNTA, "usuario", texto=texto)

    def limite(self, turno: int, texto: str) -> None:
        self._nuevo(turno, TipoPaso.LIMITE, "sistema", texto=texto)

    def actualizacion(self, turno: int, nodo: str, cambios: dict[str, Any]) -> None:
        if "resumen" in cambios:
            eliminados = len(cambios.get("messages", []))
            self._nuevo(
                turno,
                TipoPaso.MEMORIA,
                nodo,
                texto=cambios["resumen"],
                resultado={"mensajes_resumidos": eliminados},
            )
            return
        mensajes: list[BaseMessage] = cambios.get("messages", [])
        for m in mensajes:
            if isinstance(m, AIMessage) and m.tool_calls:
                if pensamiento := texto_de(m).strip():
                    self._nuevo(turno, TipoPaso.PENSAMIENTO, nodo, texto=pensamiento)
                for llamada in m.tool_calls:
                    self._nuevo(
                        turno,
                        TipoPaso.ACCION,
                        nodo,
                        herramienta=llamada["name"],
                        argumentos=dict(llamada["args"]),
                    )
            elif isinstance(m, AIMessage):
                self._nuevo(turno, TipoPaso.RESPUESTA, nodo, texto=texto_de(m))
            elif isinstance(m, ToolMessage):
                resultado = _a_json(texto_de(m))
                es_error = m.status == "error" or (
                    isinstance(resultado, dict) and "error" in resultado
                )
                self._nuevo(
                    turno,
                    TipoPaso.OBSERVACION,
                    nodo,
                    herramienta=m.name,
                    resultado=resultado,
                    es_error=es_error,
                )

    def resumen_por_turno(self) -> dict[int, dict[str, int]]:
        resumen: dict[int, dict[str, int]] = {}
        for p in self.pasos:
            fila = resumen.setdefault(p.turno, {"acciones": 0, "errores": 0})
            if p.tipo is TipoPaso.ACCION:
                fila["acciones"] += 1
            if p.es_error:
                fila["errores"] += 1
        return resumen

    def a_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "metadatos": self.metadatos,
            "resumen_por_turno": {str(t): v for t, v in self.resumen_por_turno().items()},
            "pasos": [asdict(p) for p in self.pasos],
        }

    def a_texto(self) -> str:
        lineas: list[str] = []
        for p in self.pasos:
            prefijo = f"[T{p.turno} #{p.n:02d}]"
            match p.tipo:
                case TipoPaso.PREGUNTA:
                    lineas.append(f"\n{prefijo} USUARIO      : {p.texto}")
                case TipoPaso.PENSAMIENTO:
                    lineas.append(f"{prefijo} PENSAMIENTO  : {p.texto}")
                case TipoPaso.ACCION:
                    args = ", ".join(f"{k}={v!r}" for k, v in (p.argumentos or {}).items())
                    lineas.append(f"{prefijo} ACCIÓN       : {p.herramienta}({args})")
                case TipoPaso.OBSERVACION:
                    marca = "OBSERVACIÓN!" if p.es_error else "OBSERVACIÓN "
                    resultado = json.dumps(p.resultado, ensure_ascii=False)
                    lineas.append(f"{prefijo} {marca} : {p.herramienta} -> {resultado}")
                case TipoPaso.MEMORIA:
                    n = (p.resultado or {}).get("mensajes_resumidos", 0)
                    lineas.append(f"{prefijo} MEMORIA      : {n} mensajes viejos -> resumen")
                case TipoPaso.RESPUESTA:
                    lineas.append(f"{prefijo} RESPUESTA    : {p.texto}")
                case TipoPaso.LIMITE:
                    lineas.append(f"{prefijo} LÍMITE       : {p.texto}")
        return "\n".join(lineas).lstrip("\n")

    def guardar(self, ruta_json: Path) -> Path:
        """Guarda la traza como .json y, al lado, una versión legible .log."""
        ruta_json.parent.mkdir(parents=True, exist_ok=True)
        ruta_json.write_text(
            json.dumps(self.a_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        ruta_log = ruta_json.with_suffix(".log")
        meta = json.dumps(self.metadatos, ensure_ascii=False)
        encabezado = f"thread_id={self.thread_id} | {meta}"
        ruta_log.write_text(f"{encabezado}\n\n{self.a_texto()}\n", encoding="utf-8")
        return ruta_log
