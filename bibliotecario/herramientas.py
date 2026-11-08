"""Herramientas del agente.

El LLM elige qué herramienta usar leyendo SOLO el docstring, por eso cada uno explica
cuándo usarla, cuándo no, qué devuelve y qué hacer ante un error. Los errores de negocio
se devuelven como datos (campo "error" + "sugerencia") en lugar de excepciones, para que
el agente pueda reintentar o pedir una aclaración.
"""

from typing import Any

from langchain_core.tools import BaseTool, tool

from bibliotecario import catalogo


@tool(parse_docstring=True)
async def buscar_libro(consulta: str) -> dict[str, Any]:
    """Busca libros del catálogo por título o autor y dice cuántos ejemplares hay libres.

    Usá esta herramienta cuando el usuario pregunte si un libro está disponible, si la
    biblioteca tiene un título o qué libros hay de un autor. Acepta texto parcial y no
    distingue mayúsculas ni tildes ("garcia marquez", "aleph").
    Devuelve una lista "resultados" con libro_id, titulo, autor, ejemplares_totales,
    disponibles y, si no hay ejemplares libres, "proxima_devolucion" (fecha ISO).
    Si no hay coincidencias devuelve "error" y una lista "quizas_quisiste_decir": si
    alguna sugerencia es claramente lo que el usuario quiso escribir, VOLVÉ A LLAMAR a
    esta herramienta con esa sugerencia; si no, preguntale al usuario.
    Si hay varios resultados y la pregunta era sobre un libro puntual, mostralos y
    pedile al usuario que elija.

    Args:
        consulta: Título o autor, completo o parcial. Ejemplos: "Rayuela", "Borges".
    """
    libros = await catalogo.buscar_libros(consulta)
    if not libros:
        return {
            "error": "SIN_RESULTADOS",
            "detalle": f"Ningún título ni autor contiene '{consulta}'.",
            "quizas_quisiste_decir": await catalogo.sugerir_titulos(consulta),
            "sugerencia": "Reintentá con una de las sugerencias o pedí aclaración al usuario.",
        }
    resultados: list[dict[str, Any]] = []
    for libro in libros:
        prestamos = await catalogo.prestamos_activos_de_libro(libro.libro_id)
        disponibles = libro.ejemplares - len(prestamos)
        fila: dict[str, Any] = {
            "libro_id": libro.libro_id,
            "titulo": libro.titulo,
            "autor": libro.autor,
            "ejemplares_totales": libro.ejemplares,
            "disponibles": disponibles,
        }
        if disponibles == 0:
            fila["proxima_devolucion"] = min(p.vence for p in prestamos).isoformat()
        resultados.append(fila)
    return {"cantidad": len(resultados), "resultados": resultados}


@tool(parse_docstring=True)
async def buscar_socio(nombre: str) -> dict[str, Any]:
    """Busca socios de la biblioteca por nombre o apellido y devuelve su socio_id.

    Usá esta herramienta cuando el usuario se identifique o mencione a un socio por su
    nombre ("soy Ana Torres", "los préstamos de Diego") y todavía no conozcas su
    socio_id. El socio_id que devuelve es lo que necesitan prestamos_de_socio y
    calcular_multa. No la uses si ya tenés el socio_id (por ejemplo, de un turno anterior).
    Si hay MÁS DE UNA coincidencia, NO elijas vos: listá las opciones y preguntale al
    usuario a cuál se refiere. Si no hay ninguna, devuelve "error".

    Args:
        nombre: Nombre y/o apellido del socio, por ejemplo "Ana Torres" o "Pérez".
    """
    socios = await catalogo.buscar_socios(nombre)
    if not socios:
        return {
            "error": "SOCIO_NO_ENCONTRADO",
            "detalle": f"No hay socios cuyo nombre contenga '{nombre}'.",
            "sugerencia": "Pedile al usuario que verifique el nombre o que indique su socio_id.",
        }
    coincidencias = [
        {"socio_id": s.socio_id, "nombre": s.nombre, "categoria": s.categoria} for s in socios
    ]
    respuesta: dict[str, Any] = {"cantidad": len(socios), "coincidencias": coincidencias}
    if len(socios) > 1:
        respuesta["ambiguo"] = True
        respuesta["sugerencia"] = "Hay varias personas: preguntá al usuario a cuál se refiere."
    return respuesta


@tool(parse_docstring=True)
async def prestamos_de_socio(socio_id: int) -> dict[str, Any]:
    """Lista los libros que un socio tiene prestados, con fecha de vencimiento y atraso.

    Usá esta herramienta cuando el usuario pregunte qué libros tiene (o tiene un socio),
    cuándo tiene que devolverlos o si alguno está vencido. Requiere el socio_id numérico:
    si solo sabés el nombre, llamá antes a buscar_socio.
    Devuelve "prestamos" ordenados por vencimiento, cada uno con libro_id, titulo,
    desde, vence y dias_atraso (0 si está al día). No calcula la multa: para eso usá
    calcular_multa. Si el socio no existe devuelve "error".

    Args:
        socio_id: Número de socio, por ejemplo 11.
    """
    socio = await catalogo.obtener_socio(socio_id)
    if socio is None:
        return {
            "error": "SOCIO_NO_ENCONTRADO",
            "detalle": f"No existe el socio {socio_id}.",
            "sugerencia": "Buscá al socio por nombre con buscar_socio o pedí el número correcto.",
        }
    prestamos: list[dict[str, Any]] = []
    for p in await catalogo.prestamos_de_socio(socio_id):
        libro = await catalogo.obtener_libro(p.libro_id)
        prestamos.append(
            {
                "libro_id": p.libro_id,
                "titulo": libro.titulo if libro else p.libro_id,
                "desde": p.desde.isoformat(),
                "vence": p.vence.isoformat(),
                "dias_atraso": p.dias_atraso,
            }
        )
    return {
        "socio_id": socio.socio_id,
        "socio": socio.nombre,
        "fecha_consulta": catalogo.HOY.isoformat(),
        "cantidad": len(prestamos),
        "prestamos": prestamos,
    }


@tool(parse_docstring=True)
async def calcular_multa(socio_id: int) -> dict[str, Any]:
    """Calcula la multa que debe pagar un socio por devolver libros fuera de término.

    Usá esta herramienta cuando el usuario pregunte si debe plata, si tiene una multa,
    cuánto tiene que pagar o si puede pedir otro libro (un socio con multa no puede
    pedir préstamos nuevos). Regla: $150 por día de atraso por cada libro; los socios
    de categoría "docente" están exentos. Requiere el socio_id numérico.
    Devuelve "total", "puede_pedir_prestamos" y el "detalle" por libro.

    Args:
        socio_id: Número de socio, por ejemplo 11.
    """
    socio = await catalogo.obtener_socio(socio_id)
    if socio is None:
        return {
            "error": "SOCIO_NO_ENCONTRADO",
            "detalle": f"No existe el socio {socio_id}.",
            "sugerencia": "Buscá al socio por nombre con buscar_socio o pedí el número correcto.",
        }
    exento = socio.categoria == "docente"
    detalle: list[dict[str, Any]] = []
    for p in await catalogo.prestamos_de_socio(socio_id):
        if p.dias_atraso == 0:
            continue
        importe = 0 if exento else p.dias_atraso * catalogo.MULTA_POR_DIA
        detalle.append({"libro_id": p.libro_id, "dias_atraso": p.dias_atraso, "importe": importe})
    total = sum(d["importe"] for d in detalle)
    return {
        "socio_id": socio.socio_id,
        "socio": socio.nombre,
        "exento": exento,
        "total": total,
        "puede_pedir_prestamos": total == 0,
        "detalle": detalle,
    }


HERRAMIENTAS: list[BaseTool] = [buscar_libro, buscar_socio, prestamos_de_socio, calcular_multa]
