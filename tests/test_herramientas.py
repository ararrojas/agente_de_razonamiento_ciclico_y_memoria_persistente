import pytest

from bibliotecario.herramientas import (
    HERRAMIENTAS,
    buscar_libro,
    buscar_socio,
    calcular_multa,
    prestamos_de_socio,
)


@pytest.mark.parametrize("herramienta", HERRAMIENTAS, ids=lambda h: h.name)
def test_docstrings_descriptivos(herramienta) -> None:  # type: ignore[no-untyped-def]
    # El LLM elige la herramienta solo por la descripción: tiene que ser detallada.
    assert len(herramienta.description) > 250
    assert "Usá esta herramienta" in herramienta.description
    for nombre, esquema in herramienta.args.items():
        assert esquema.get("description"), f"falta documentar el argumento {nombre}"


async def test_buscar_libro_sin_tildes_y_disponibilidad() -> None:
    r = await buscar_libro.ainvoke({"consulta": "garcia marquez"})
    titulos = {x["titulo"]: x["disponibles"] for x in r["resultados"]}
    assert titulos == {"Cien años de soledad": 2, "El amor en los tiempos del cólera": 1}


async def test_buscar_libro_agotado_informa_devolucion() -> None:
    r = await buscar_libro.ainvoke({"consulta": "Rayuela"})
    assert r["resultados"][0]["disponibles"] == 0
    assert r["resultados"][0]["proxima_devolucion"] == "2026-10-15"


async def test_buscar_libro_con_error_de_tipeo_sugiere() -> None:
    r = await buscar_libro.ainvoke({"consulta": "Cien años de solead"})
    assert r["error"] == "SIN_RESULTADOS"
    assert "Cien años de soledad" in r["quizas_quisiste_decir"]


async def test_buscar_socio_ambiguo() -> None:
    r = await buscar_socio.ainvoke({"nombre": "perez"})
    assert r["ambiguo"] is True and r["cantidad"] == 2


async def test_prestamos_y_atraso() -> None:
    r = await prestamos_de_socio.ainvoke({"socio_id": 11})
    assert [p["dias_atraso"] for p in r["prestamos"]] == [22, 0]


async def test_socio_inexistente_devuelve_error() -> None:
    r = await prestamos_de_socio.ainvoke({"socio_id": 99})
    assert r["error"] == "SOCIO_NO_ENCONTRADO"


async def test_multa_y_exencion_docente() -> None:
    ana = await calcular_multa.ainvoke({"socio_id": 11})
    assert ana["total"] == 22 * 150 and ana["puede_pedir_prestamos"] is False
    diego = await calcular_multa.ainvoke({"socio_id": 14})
    assert diego["total"] == 43 * 150
    camila = await calcular_multa.ainvoke({"socio_id": 13})
    assert camila["exento"] is True and camila["total"] == 0
