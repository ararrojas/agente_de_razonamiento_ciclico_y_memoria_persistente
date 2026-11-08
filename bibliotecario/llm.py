"""Fábrica del modelo de lenguaje según el proveedor configurado."""

import warnings

from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from bibliotecario.config import Configuracion

# Algunos modelos de Gemini usan temperatura fija e ignoran temperature=0; la librería
# lo avisa en cada llamada. Es inofensivo, así que se silencia para no ensuciar la salida.
warnings.filterwarnings("ignore", message=r".*uses fixed sampling defaults.*")


def _limitador(config: Configuracion) -> InMemoryRateLimiter | None:
    """Espacia las llamadas al LLM para no superar la cuota por minuto del proveedor."""
    if config.solicitudes_por_minuto == 0:
        return None
    return InMemoryRateLimiter(
        requests_per_second=config.solicitudes_por_minuto / 60,
        check_every_n_seconds=0.5,
        max_bucket_size=1,
    )


def crear_llm(config: Configuracion) -> BaseChatModel:
    limitador = _limitador(config)
    if config.proveedor == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.modelo,
            google_api_key=config.api_key,
            temperature=0,
            rate_limiter=limitador,
            # Pocos reintentos y timeout: si el modelo está saturado (503) falla rápido
            # con un error claro en vez de quedar reintentando varios minutos.
            max_retries=2,
            timeout=90,
        )

    if config.proveedor == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.modelo,
            api_key=config.api_key,  # type: ignore[arg-type]
            temperature=0,
            rate_limiter=limitador,
        )

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(  # type: ignore[call-arg]
        model=config.modelo,
        api_key=config.api_key,  # type: ignore[arg-type]
        temperature=0,
        max_tokens=1024,
        rate_limiter=limitador,
    )
