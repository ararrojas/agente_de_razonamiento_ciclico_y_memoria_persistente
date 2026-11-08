"""Demo multi-turno: genera la traza ReAct en trazas/ y prueba la memoria tras un reinicio.

python -m bibliotecario.demo            # con el LLM configurado en .env
python -m bibliotecario.demo --offline  # con el modelo guionado, sin API key
"""

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from bibliotecario.config import (
    RAIZ_PROYECTO,
    Configuracion,
    ErrorDeConfiguracion,
    cargar_configuracion,
)
from bibliotecario.ejecucion import configuracion, ejecutar_turno
from bibliotecario.grafo import GrafoBiblioteca, construir_grafo
from bibliotecario.herramientas import HERRAMIENTAS
from bibliotecario.llm import crear_llm
from bibliotecario.memoria import Resumidor, resumidor_con_llm, resumidor_extractivo
from bibliotecario.persistencia import abrir_memoria
from bibliotecario.simulado import (
    PREGUNTA_TRAS_REINICIO,
    PREGUNTAS_DEMO,
    ModeloGuionado,
    guion_demo,
)
from bibliotecario.traza import TrazaReAct

SALIDA_POR_DEFECTO = RAIZ_PROYECTO / "trazas" / "traza_react.json"


def _argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo del agente bibliotecario")
    p.add_argument("--offline", action="store_true", help="usa el modelo guionado (sin API)")
    p.add_argument("--thread-id", default=f"demo-{datetime.now():%Y%m%d-%H%M%S}")
    p.add_argument("--salida", type=Path, default=SALIDA_POR_DEFECTO)
    p.add_argument(
        "--resumen-simple",
        action="store_true",
        help="resume la memoria sin LLM (ahorra llamadas en planes gratuitos)",
    )
    return p.parse_args()


def _grafo(
    llm: BaseChatModel, resumidor: Resumidor, config: Configuracion, saver: object
) -> GrafoBiblioteca:
    return construir_grafo(
        llm,
        HERRAMIENTAS,
        saver,  # type: ignore[arg-type]
        resumidor,
        config.umbral_resumen,
        config.turnos_a_conservar,
    )


async def _demo(args: argparse.Namespace) -> None:
    config = cargar_configuracion(requiere_api_key=not args.offline)
    llm: BaseChatModel
    resumidor: Resumidor
    if args.offline:
        llm = ModeloGuionado(respuestas=guion_demo())
        resumidor = resumidor_extractivo
        origen = "offline: modelo guionado (herramientas, grafo y SQLite reales)"
    else:
        llm = crear_llm(config)
        resumidor = resumidor_extractivo if args.resumen_simple else resumidor_con_llm(llm)
        origen = f"{config.proveedor}:{config.modelo}"

    traza = TrazaReAct(
        args.thread_id,
        {
            "modelo": origen,
            "recursion_limit": config.limite_recursion,
            "umbral_resumen": config.umbral_resumen,
            "generada": datetime.now(UTC).isoformat(timespec="seconds"),
            "completa": False,
        },
    )
    try:
        await _correr_turnos(args, config, llm, resumidor, traza)
        traza.metadatos["completa"] = True
    except Exception as error:
        # Se guarda lo que se alcanzó a registrar antes del error (por ejemplo, cuota agotada).
        traza.metadatos["error"] = f"{type(error).__name__}: {str(error)[:300]}"
        raise
    finally:
        ruta_log = traza.guardar(args.salida)
        print(f"\nTraza guardada en {args.salida} y {ruta_log}")


async def _correr_turnos(
    args: argparse.Namespace,
    config: Configuracion,
    llm: BaseChatModel,
    resumidor: Resumidor,
    traza: TrazaReAct,
) -> None:
    cfg = configuracion(args.thread_id, config.limite_recursion)

    # Sesión 1: varios turnos con el mismo thread_id.
    async with abrir_memoria(config.ruta_checkpoints) as saver:
        grafo = _grafo(llm, resumidor, config, saver)
        for turno, pregunta in enumerate(PREGUNTAS_DEMO, start=1):
            respuesta = await ejecutar_turno(
                grafo, pregunta, args.thread_id, config.limite_recursion, turno, traza
            )
            print(f"\n[T{turno}] Usuario: {pregunta}\n     Biblio: {respuesta}")
            traza.guardar(args.salida)  # guardado parcial después de cada turno

    # Sesión 2: conexión nueva a la misma base (como si el proceso se hubiera reiniciado).
    print("\n--- Se cierra la conexión y se abre otra a la misma base SQLite ---")
    async with abrir_memoria(config.ruta_checkpoints) as saver:
        grafo = _grafo(llm, resumidor, config, saver)
        turno = len(PREGUNTAS_DEMO) + 1
        respuesta = await ejecutar_turno(
            grafo, PREGUNTA_TRAS_REINICIO, args.thread_id, config.limite_recursion, turno, traza
        )
        print(f"\n[T{turno}] Usuario: {PREGUNTA_TRAS_REINICIO}\n     Biblio: {respuesta}")
        estado = (await grafo.aget_state(cfg)).values

    print("\n=== Traza ReAct ===\n" + traza.a_texto())
    print(
        f"\nthread_id={args.thread_id} | mensajes en estado={len(estado['messages'])} | "
        f"herramientas pedidas={estado['llamadas_herramientas']} | "
        f"compactaciones de memoria={estado.get('compactaciones', 0)}"
    )


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(_demo(_argumentos()))
    except ErrorDeConfiguracion as e:
        sys.exit(f"Error de configuración: {e}")


if __name__ == "__main__":
    main()
