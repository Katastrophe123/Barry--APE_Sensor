"""Concrete :class:`~app.llm.base.LLMClient` implementations."""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.llm.base import (
    LLMClient,
    LLMPermanentError,
    LLMResponse,
    LLMTransientError,
)
from app.logging_config import get_logger

logger = get_logger(__name__)

_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _classify_status(
    status: int, body: str, provider: str
) -> LLMTransientError | LLMPermanentError:
    """Map an HTTP status onto a transient/permanent LLM error."""
    message = f"{provider} returned HTTP {status}: {body[:300]}"
    if status in _TRANSIENT_STATUS:
        return LLMTransientError(message)
    return LLMPermanentError(message)


class OpenAICompatibleClient(LLMClient):
    """Client for any ``/chat/completions`` endpoint (OpenAI, vLLM, Groq, ...)."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    async def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        response = await self._client.post("/chat/completions", json=payload)
        if response.status_code >= 400:
            raise _classify_status(response.status_code, response.text, self.name)

        try:
            data: dict[str, Any] = response.json()
            text = data["choices"][0]["message"]["content"] or ""
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMTransientError(f"Malformed response from {self.name}: {exc}") from exc

        return LLMResponse(text=text, model=data.get("model", self.model), raw=data)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class HuggingFaceInferenceClient(LLMClient):
    """Client for the Hugging Face Inference API text-generation task."""

    name = "huggingface"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api-inference.huggingface.co/models",
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    @staticmethod
    def _build_prompt(system: str, user: str) -> str:
        """Render Llama-3 style chat markers around the two messages."""
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    async def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "inputs": self._build_prompt(system, user),
            "parameters": {
                "max_new_tokens": max_tokens,
                # HF rejects temperature == 0; nudge to a near-greedy value.
                "temperature": max(temperature, 0.01),
                "return_full_text": False,
            },
            "options": {"wait_for_model": True},
        }
        response = await self._client.post(f"/{self.model}", json=payload)
        if response.status_code >= 400:
            raise _classify_status(response.status_code, response.text, self.name)

        try:
            data = response.json()
            text = data[0]["generated_text"] if isinstance(data, list) else data["generated_text"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMTransientError(f"Malformed response from {self.name}: {exc}") from exc

        return LLMResponse(
            text=text,
            model=self.model,
            raw=data if isinstance(data, dict) else {"data": data},
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


# --------------------------------------------------------------------------- #
# Offline stub
# --------------------------------------------------------------------------- #
class StubLLMClient(LLMClient):
    """Deterministic, offline stand-in that writes a usable maintenance note.

    Exists so ``docker compose up`` and the test suite work with no API key.
    The note is assembled from the numbers in the prompt rather than generated,
    which is exactly what the production fallback does too - see
    :class:`~app.services.insight_service.MaintenanceNoteService`.
    """

    name = "stub"

    def __init__(self, *, model: str = "stub-notes-v1") -> None:
        self.model = model

    #: Band -> note. Keyed off the prompt so the stub never recommends a
    #: workshop visit for a vehicle it was told is low risk.
    _NOTES = {
        "high": (
            "Book this vehicle in for an air pressure system inspection before its "
            "next long run. The elevated score makes a pre-emptive check cheaper "
            "than a roadside failure. Confirm compressor output and the system "
            "leak-down rate, and review the sensor readings listed above."
        ),
        "medium": (
            "Schedule an air pressure system check at the next depot visit. The "
            "score is elevated but not urgent, so there is no need to pull the "
            "vehicle off its route. Confirm the air dryer is cycling and look for "
            "any slow pressure loss."
        ),
        "low": (
            "No air pressure system action is needed beyond the normal service "
            "schedule. The score is below the alerting threshold for this model. "
            "Keep the vehicle on its usual maintenance interval."
        ),
    }

    @staticmethod
    def _detect_band(user: str) -> str:
        for line in user.splitlines():
            if line.lower().startswith("risk band:"):
                return line.split(":", 1)[1].strip().lower()
        return "medium"

    async def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> LLMResponse:
        band = self._detect_band(user)
        text = self._NOTES.get(band, self._NOTES["medium"])
        return LLMResponse(
            text=text,
            model=self.model,
            raw={"stub": True, "band": band, "prompt_chars": len(user)},
        )
