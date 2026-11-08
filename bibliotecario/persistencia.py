"""Checkpointer SQLite: guarda el estado del grafo después de cada paso, por thread_id."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


@asynccontextmanager
async def abrir_memoria(ruta_db: Path) -> AsyncIterator[AsyncSqliteSaver]:
    """Abre (o crea) la base SQLite de checkpoints.

    Se usa AsyncSqliteSaver, la variante asíncrona de SqliteSaver (mismo paquete
    langgraph-checkpoint-sqlite y mismo formato de tablas), porque el grafo se ejecuta
    con asyncio (astream). SqliteSaver síncrono bloquearía el event loop.
    """
    ruta_db.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(ruta_db)) as saver:
        yield saver
