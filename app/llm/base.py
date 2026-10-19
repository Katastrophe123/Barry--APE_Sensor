"""Provider-agnostic LLM client contract.

The LLM is a *peripheral* here, not the product: it turns a high-risk numeric
prediction into a sentence an operator can act on. Every call goes through
:class:`LLMClient`, so the provider is a configuration change, and the service
degrades to a deterministic template if the model is unavailable - a prediction
must never be lost because a text generator was down.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class LLMError(Exception):
    """Base class for all LLM failures."""


class LLMTransientError(LLMError):
    """Failure that is worth retrying (timeout, 5xx, rate limit, bad JSON)."""


class LLMPermanentError(LLMError):
    """Failure that retrying cannot fix (bad credentials, unknown model)."""


class LLMInvalidResponseError(LLMTransientError):
    """The model replied, but not with the structure we asked for."""


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    raw: dict[str, Any] = field(default_factory=dict)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort extraction of a single JSON object from model output.

    Handles the three things models actually do: return clean JSON, wrap it in a
    markdown fence, or surround it with prose. Raises
    :class:`LLMInvalidResponseError` when no object can be recovered, which the
    retry wrapper treats as transient.
    """
    candidates: list[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise LLMInvalidResponseError(
        f"Expected a JSON object in the model response, got: {text[:300]!r}"
    )


class LLMClient(ABC):
    """Minimal surface the agent needs from a language model."""

    name: str = "llm"
    """Provider identifier, e.g. ``openai``."""

    model: str = "unknown"
    """Concrete model identifier, surfaced in logs and the eval report."""

    @abstractmethod
    async def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> LLMResponse:
        """Return a completion for the given system/user pair."""

    async def generate_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> dict[str, Any]:
        """Call :meth:`generate` and parse the reply as a JSON object."""
        response = await self.generate(
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return extract_json_object(response.text)

    async def aclose(self) -> None:
        """Release any underlying network resources."""
        return
