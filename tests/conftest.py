from pathlib import Path

import pytest

import bibliotecario.catalogo as catalogo


@pytest.fixture(autouse=True)
def sin_latencia(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(catalogo, "LATENCIA_SEGUNDOS", 0)


@pytest.fixture
def ruta_db(tmp_path: Path) -> Path:
    return tmp_path / "memoria.sqlite"
