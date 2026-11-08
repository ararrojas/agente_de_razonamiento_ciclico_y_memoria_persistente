"""Tests del grafo con el modelo guionado: sin llamadas a APIs, pero con herramientas,
ToolNode, reducers y SQLite reales."""

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from bibliotecario.ejecucion import configuracion, ejecutar_turno, mensaje_de_limite
from bibliotecario.grafo import GrafoBiblioteca, construir_grafo
from bibliotecario.herramientas import HERRAMIENTAS
from bibliotecario.memoria import punto_de_corte, resumidor_extractivo
from bibliotecario.persistencia import abrir_memoria
from bibliotecario.simulado import ModeloGuionado, accion
from bibliotecario.traza import TipoPaso, TrazaReAct

LIMITE = 10


def _grafo(modelo: ModeloGuionado, saver: object, umbral: int = 100) -> GrafoBiblioteca:
    return construir_grafo(
        modelo,
        HERRAMIENTAS,
        saver,
        resumidor_extractivo,
        umbral,
        1,  # type: ignore[arg-type]
    )


async def test_encadena_herramientas_y_suma_contador(ruta_db: Path) -> None:
    modelo = ModeloGuionado(
        respuestas=[
            accion("buscar_socio", {"nombre": "Ana"}, "a"),
            accion("calcular_multa", {"socio_id": 11}, "b"),
            AIMessage(content="Debés $3.300."),
        ]
    )
    traza = TrazaReAct("t")
    async with abrir_memoria(ruta_db) as saver:
        grafo = _grafo(modelo, saver)
        respuesta = await ejecutar_turno(grafo, "¿Cuánto debe Ana?", "t", LIMITE, 1, traza)
        estado = (await grafo.aget_state(configuracion("t", LIMITE))).values

    assert respuesta == "Debés $3.300."
    assert estado["llamadas_herramientas"] == 2  # reducer operator.add
    tools = [m for m in estado["messages"] if isinstance(m, ToolMessage)]
    assert '"total": 3300' in str(tools[1].content)
    assert [p.tipo for p in traza.pasos].count(TipoPaso.ACCION) == 2


async def test_reintenta_tras_error_de_herramienta(ruta_db: Path) -> None:
    modelo = ModeloGuionado(
        respuestas=[
            accion("buscar_libro", {"consulta": "Ficcions"}, "a"),
            accion("buscar_libro", {"consulta": "Ficciones"}, "b"),
            AIMessage(content="Hay 2 ejemplares de Ficciones."),
        ]
    )
    traza = TrazaReAct("t")
    async with abrir_memoria(ruta_db) as saver:
        await ejecutar_turno(_grafo(modelo, saver), "¿Está Ficcions?", "t", LIMITE, 1, traza)

    observaciones = [p for p in traza.pasos if p.tipo is TipoPaso.OBSERVACION]
    assert [o.es_error for o in observaciones] == [True, False]
    # En el segundo intento el modelo vio el error y la sugerencia de la herramienta.
    visto = modelo.recibido[1][-1]
    assert isinstance(visto, ToolMessage) and "Ficciones" in str(visto.content)


async def test_argumentos_invalidos_vuelven_al_modelo(ruta_db: Path) -> None:
    modelo = ModeloGuionado(
        respuestas=[
            accion("prestamos_de_socio", {"socio_id": "once"}, "a"),
            accion("prestamos_de_socio", {"socio_id": 11}, "b"),
            AIMessage(content="ok"),
        ]
    )
    async with abrir_memoria(ruta_db) as saver:
        await ejecutar_turno(_grafo(modelo, saver), "préstamos del 11", "t", LIMITE)
    error = modelo.recibido[1][-1]
    assert isinstance(error, ToolMessage) and error.status == "error"


async def test_memoria_mismo_thread_y_tras_reabrir_la_base(ruta_db: Path) -> None:
    modelo = ModeloGuionado(
        respuestas=[
            AIMessage(content="¡Hola Ana!"),
            AIMessage(content="Sos Ana."),
            AIMessage(content="No sé quién sos."),
        ]
    )
    async with abrir_memoria(ruta_db) as saver:
        await ejecutar_turno(_grafo(modelo, saver), "Soy Ana", "ana", LIMITE)

    async with abrir_memoria(ruta_db) as saver:  # conexión nueva = proceso reiniciado
        grafo = _grafo(modelo, saver)
        await ejecutar_turno(grafo, "¿Quién soy?", "ana", LIMITE)
        await ejecutar_turno(grafo, "¿Quién soy?", "otro-thread", LIMITE)

    contexto_ana = [m.content for m in modelo.recibido[1] if isinstance(m, HumanMessage)]
    contexto_otro = [m.content for m in modelo.recibido[2] if isinstance(m, HumanMessage)]
    assert contexto_ana == ["Soy Ana", "¿Quién soy?"]
    assert contexto_otro == ["¿Quién soy?"]


async def test_limite_de_recursion_corta_y_deja_estado_valido(ruta_db: Path) -> None:
    bucle = [accion("buscar_libro", {"consulta": "Borges"}, f"c{i}") for i in range(10)]
    modelo = ModeloGuionado(respuestas=[*bucle, AIMessage(content="sigo vivo")])
    async with abrir_memoria(ruta_db) as saver:
        grafo = _grafo(modelo, saver)
        respuesta = await ejecutar_turno(grafo, "loop", "t", 6)
        assert respuesta == mensaje_de_limite(6)
        # Todas las tool_calls quedaron respondidas: el próximo turno funciona.
        modelo.posicion = len(bucle)
        assert await ejecutar_turno(grafo, "hola", "t", 6) == "sigo vivo"
        mensajes = (await grafo.aget_state(configuracion("t", 6))).values["messages"]
    pedidas = {c["id"] for m in mensajes if isinstance(m, AIMessage) for c in m.tool_calls}
    respondidas = {m.tool_call_id for m in mensajes if isinstance(m, ToolMessage)}
    assert pedidas == respondidas


async def test_compacta_memoria_sin_romper_pares_tool_call(ruta_db: Path) -> None:
    respuestas: list[AIMessage] = []
    for i in range(4):
        respuestas += [
            accion("buscar_libro", {"consulta": "Borges"}, f"c{i}"),
            AIMessage(content=f"respuesta {i}"),
        ]
    modelo = ModeloGuionado(respuestas=respuestas)
    async with abrir_memoria(ruta_db) as saver:
        grafo = _grafo(modelo, saver, umbral=6)
        for i in range(4):
            await ejecutar_turno(grafo, f"pregunta {i}", "t", LIMITE)
        estado = (await grafo.aget_state(configuracion("t", LIMITE))).values

    assert estado["compactaciones"] >= 1
    assert "pregunta 0" in estado["resumen"]
    assert isinstance(estado["messages"][0], HumanMessage)
    # El modelo recibe el resumen en el prompt de sistema.
    assert "Resumen de la conversación anterior" in str(modelo.recibido[-1][0].content)


def test_punto_de_corte_respeta_turnos() -> None:
    msgs = [HumanMessage("a"), AIMessage("b"), HumanMessage("c"), AIMessage("d")]
    assert punto_de_corte(msgs, 1) == 2
    assert punto_de_corte(msgs, 2) == 0
