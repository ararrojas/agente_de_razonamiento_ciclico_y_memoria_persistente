"""Configuración leída desde variables de entorno (archivo .env)."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ_PROYECTO / ".env")

type Proveedor = Literal["google", "anthropic", "openai"]

MODELO_POR_DEFECTO: dict[Proveedor, str] = {
    "google": "gemini-3.8-flash",
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4o-mini",
}
VARIABLE_CLAVE: dict[Proveedor, str] = {
    "google": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


class ErrorDeConfiguracion(RuntimeError):
    """Falta una variable de entorno o tiene un valor inválido."""


@dataclass(frozen=True, slots=True)
class Configuracion:
    proveedor: Proveedor
    modelo: str
    api_key: str
    ruta_checkpoints: Path
    limite_recursion: int
    umbral_resumen: int
    turnos_a_conservar: int
    # Solicitudes por minuto al LLM (0 = sin límite). Útil en planes gratuitos.
    solicitudes_por_minuto: int = 0


def _proveedor() -> Proveedor:
    valor = os.getenv("LLM_PROVIDER", "google").strip().lower()
    match valor:
        case "google" | "gemini":
            return "google"
        case "anthropic":
            return "anthropic"
        case "openai":
            return "openai"
        case _:
            raise ErrorDeConfiguracion(
                f"LLM_PROVIDER='{valor}' no es válido: usá 'google', 'anthropic' u 'openai'."
            )


def _api_key(proveedor: Proveedor) -> str:
    variable = VARIABLE_CLAVE[proveedor]
    valor = os.getenv(variable, "").strip()
    if not valor or valor.startswith("tu-clave"):
        raise ErrorDeConfiguracion(f"Completá {variable} en {RAIZ_PROYECTO / '.env'}.")
    return valor


def _entero(variable: str, por_defecto: int, minimo: int = 1) -> int:
    texto = os.getenv(variable, "").strip() or str(por_defecto)
    try:
        valor = int(texto)
    except ValueError as exc:
        raise ErrorDeConfiguracion(f"{variable} debe ser un número entero ('{texto}').") from exc
    if valor < minimo:
        raise ErrorDeConfiguracion(f"{variable} debe ser >= {minimo}.")
    return valor


def cargar_configuracion(requiere_api_key: bool = True) -> Configuracion:
    proveedor = _proveedor()
    ruta = Path(os.getenv("CHECKPOINT_DB", "").strip() or "data/memoria.sqlite")
    if not ruta.is_absolute():
        ruta = RAIZ_PROYECTO / ruta
    return Configuracion(
        proveedor=proveedor,
        modelo=os.getenv("LLM_MODEL", "").strip() or MODELO_POR_DEFECTO[proveedor],
        api_key=_api_key(proveedor) if requiere_api_key else "",
        ruta_checkpoints=ruta,
        limite_recursion=_entero("RECURSION_LIMIT", 10, minimo=3),
        umbral_resumen=_entero("UMBRAL_RESUMEN", 14, minimo=4),
        turnos_a_conservar=_entero("TURNOS_A_CONSERVAR", 2),
        solicitudes_por_minuto=_entero("LLM_RPM", 0, minimo=0),
    )
