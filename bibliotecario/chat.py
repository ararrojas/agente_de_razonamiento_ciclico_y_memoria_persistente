"""Chat por consola con memoria persistente.

python -m bibliotecario.chat --thread-id ana
python -m bibliotecario.chat --thread-id ana --pregunta "¿Qué libros tengo?" --traza
"""

import argparse
import asyncio
import logging
import sys

from langchain_core.messages import AIMessage, HumanMessage

from bibliotecario.config import ErrorDeConfiguracion, cargar_configuracion
from bibliotecario.ejecucion import configuracion, ejecutar_turno
from bibliotecario.grafo import construir_grafo
from bibliotecario.herramientas import HERRAMIENTAS
from bibliotecario.llm import crear_llm
from bibliotecario.memoria import resumidor_con_llm, texto_de
from bibliotecario.persistencia import abrir_memoria
from bibliotecario.traza import TrazaReAct

SALIR = {"salir", "chau", "exit", "quit"}


def _argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chat con el agente bibliotecario")
    p.add_argument("--thread-id", default="principal", help="identificador de la conversación")
    p.add_argument("--pregunta", help="hace una sola pregunta y termina")
    p.add_argument("--traza", action="store_true", help="muestra el ciclo ReAct de cada turno")
    p.add_argument("--historial", action="store_true", help="muestra lo que recuerda y termina")
    return p.parse_args()


async def _chat(args: argparse.Namespace) -> None:
    config = cargar_configuracion()
    llm = crear_llm(config)
    async with abrir_memoria(config.ruta_checkpoints) as saver:
        grafo = construir_grafo(
            llm,
            HERRAMIENTAS,
            saver,
            resumidor_con_llm(llm),
            config.umbral_resumen,
            config.turnos_a_conservar,
        )
        estado = (await grafo.aget_state(configuracion(args.thread_id, 1))).values
        previos = estado.get("messages", [])
        turno = sum(isinstance(m, HumanMessage) for m in previos)

        if args.historial:
            if resumen := estado.get("resumen"):
                print(f"Resumen:\n{resumen}\n")
            for m in previos:
                if isinstance(m, HumanMessage):
                    print(f"Vos: {texto_de(m)}")
                elif isinstance(m, AIMessage) and not m.tool_calls:
                    print(f"Biblio: {texto_de(m)}")
            return

        print(
            f"thread_id={args.thread_id} | {config.proveedor}:{config.modelo} | "
            f"{len(previos)} mensajes en memoria. Escribí 'salir' para terminar."
        )
        while True:
            if args.pregunta:
                pregunta = args.pregunta
            else:
                pregunta = (await asyncio.to_thread(input, "\nVos: ")).strip()
                if pregunta.lower() in SALIR:
                    break
                if not pregunta:
                    continue
            turno += 1
            traza = TrazaReAct(args.thread_id)
            respuesta = await ejecutar_turno(
                grafo, pregunta, args.thread_id, config.limite_recursion, turno, traza
            )
            if args.traza:
                print(traza.a_texto())
            print(f"\nBiblio: {respuesta}")
            if args.pregunta:
                break


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(_chat(_argumentos()))
    except ErrorDeConfiguracion as e:
        sys.exit(f"Error de configuración: {e}")
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
