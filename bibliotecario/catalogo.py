"""Base de datos simulada de la biblioteca (libros, socios y préstamos) con acceso asíncrono."""

import asyncio
import difflib
import unicodedata
from dataclasses import dataclass
from datetime import date

# Fecha fija para que las multas y los atrasos sean reproducibles en la demo y los tests.
HOY = date(2026, 11, 6)
MULTA_POR_DIA = 150  # pesos por día de atraso y por libro
LATENCIA_SEGUNDOS = 0.02


@dataclass(frozen=True, slots=True)
class Libro:
    libro_id: str
    titulo: str
    autor: str
    ejemplares: int


@dataclass(frozen=True, slots=True)
class Socio:
    socio_id: int
    nombre: str
    categoria: str


@dataclass(frozen=True, slots=True)
class Prestamo:
    libro_id: str
    socio_id: int
    desde: date
    vence: date

    @property
    def dias_atraso(self) -> int:
        return max(0, (HOY - self.vence).days)


LIBROS: tuple[Libro, ...] = (
    Libro("L-01", "Rayuela", "Julio Cortázar", 2),
    Libro("L-02", "Cien años de soledad", "Gabriel García Márquez", 3),
    Libro("L-03", "El Aleph", "Jorge Luis Borges", 1),
    Libro("L-04", "Ficciones", "Jorge Luis Borges", 2),
    Libro("L-05", "La casa de los espíritus", "Isabel Allende", 1),
    Libro("L-06", "El amor en los tiempos del cólera", "Gabriel García Márquez", 1),
)

SOCIOS: tuple[Socio, ...] = (
    Socio(11, "Ana Torres", "estudiante"),
    Socio(12, "Bruno Pérez", "general"),
    Socio(13, "Camila Pérez", "docente"),
    Socio(14, "Diego Sosa", "general"),
)

PRESTAMOS: tuple[Prestamo, ...] = (
    Prestamo("L-01", 11, date(2026, 10, 1), date(2026, 10, 15)),
    Prestamo("L-03", 11, date(2026, 10, 25), date(2026, 11, 8)),
    Prestamo("L-01", 12, date(2026, 10, 20), date(2026, 11, 3)),
    Prestamo("L-05", 13, date(2026, 11, 1), date(2026, 11, 15)),
    Prestamo("L-02", 14, date(2026, 9, 10), date(2026, 9, 24)),
)


def normalizar(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFD", texto.lower().strip())
    return "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")


async def _latencia() -> None:
    await asyncio.sleep(LATENCIA_SEGUNDOS)


async def buscar_libros(consulta: str) -> list[Libro]:
    await _latencia()
    q = normalizar(consulta)
    if not q:
        return []
    return [lb for lb in LIBROS if q in normalizar(lb.titulo) or q in normalizar(lb.autor)]


async def sugerir_titulos(consulta: str, maximo: int = 3) -> list[str]:
    await _latencia()
    candidatos = {normalizar(lb.titulo): lb.titulo for lb in LIBROS}
    candidatos |= {normalizar(lb.autor): lb.autor for lb in LIBROS}
    cercanos = difflib.get_close_matches(normalizar(consulta), candidatos, n=maximo, cutoff=0.6)
    return [candidatos[c] for c in cercanos]


async def obtener_libro(libro_id: str) -> Libro | None:
    await _latencia()
    buscado = libro_id.strip().upper()
    return next((lb for lb in LIBROS if lb.libro_id == buscado), None)


async def prestamos_activos_de_libro(libro_id: str) -> list[Prestamo]:
    await _latencia()
    return [p for p in PRESTAMOS if p.libro_id == libro_id]


async def buscar_socios(nombre: str) -> list[Socio]:
    await _latencia()
    q = normalizar(nombre)
    if not q:
        return []
    return [s for s in SOCIOS if q in normalizar(s.nombre)]


async def obtener_socio(socio_id: int) -> Socio | None:
    await _latencia()
    return next((s for s in SOCIOS if s.socio_id == socio_id), None)


async def prestamos_de_socio(socio_id: int) -> list[Prestamo]:
    await _latencia()
    return sorted((p for p in PRESTAMOS if p.socio_id == socio_id), key=lambda p: p.vence)
