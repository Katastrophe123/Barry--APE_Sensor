"""Builds the configured :class:`~app.llm.base.LLMClient`."""

from __future__ import annotations

from app.config import Settings
from app.llm.base import LLMClient, LLMPermanentError
from app.llm.providers import (
    HuggingFaceInferenceClient,
    OpenAICompatibleClient,
    StubLLMClient,
)
from app.logging_config import get_logger

logger = get_logger(__name__)


def build_llm_client(settings: Settings) -> LLMClient:
    """Instantiate the provider named by ``settings.llm_provider``.

    Raises :class:`LLMPermanentError` at startup (rather than on the first
    request) when a networked provider is selected without an API key.
    """
    provider = settings.llm_provider

    if provider == "stub":
        logger.info("llm_client_selected", provider="stub", model="stub-notes-v1")
        return StubLLMClient()

    if not settings.llm_api_key:
        raise LLMPermanentError(
            f"LLM_PROVIDER={provider} requires LLM_API_KEY to be set "
            "(use LLM_PROVIDER=stub to run without credentials)."
        )

    logger.info("llm_client_selected", provider=provider, model=settings.llm_model)
    if provider == "openai":
        return OpenAICompatibleClient(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
        )
    return HuggingFaceInferenceClient(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout=settings.llm_timeout_seconds,
    )
