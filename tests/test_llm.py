"""Tests for the swappable LLM client layer.

The LLM is optional in this service (maintenance notes only), but the provider
abstraction still has to hold: tests must never need a real API key, and a
provider swap must be configuration rather than code.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.llm.base import (
    LLMInvalidResponseError,
    LLMPermanentError,
    LLMTransientError,
    extract_json_object,
)
from app.llm.factory import build_llm_client
from app.llm.providers import (
    HuggingFaceInferenceClient,
    OpenAICompatibleClient,
    StubLLMClient,
)
from app.llm.retry import RetriesExhaustedError, call_with_retries


def openai_client(transport: httpx.MockTransport) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        api_key="test-key",
        base_url="https://llm.example/v1",
        model="test-model",
        client=httpx.AsyncClient(transport=transport, base_url="https://llm.example/v1"),
    )


class TestExtractJsonObject:
    def test_parses_clean_json(self) -> None:
        assert extract_json_object('{"ok": true}') == {"ok": True}

    def test_parses_fenced_json(self) -> None:
        assert extract_json_object('```json\n{"ok": 1}\n```') == {"ok": 1}

    def test_parses_json_in_prose(self) -> None:
        assert extract_json_object('Sure: {"ok": 2} - done') == {"ok": 2}

    @pytest.mark.parametrize("raw", ["", "no json", "[1,2]", '{"unterminated": '])
    def test_rejects_unusable_payloads(self, raw: str) -> None:
        with pytest.raises(LLMInvalidResponseError):
            extract_json_object(raw)


class TestRetryWrapper:
    async def test_returns_on_success(self, fake_sleep: Any, sleep_calls: list[float]) -> None:
        async def operation() -> str:
            return "ok"

        assert await call_with_retries(operation, max_attempts=2, sleep=fake_sleep) == "ok"
        assert sleep_calls == []

    async def test_backoff_is_exponential(self, fake_sleep: Any, sleep_calls: list[float]) -> None:
        async def operation() -> str:
            raise LLMTransientError("down")

        with pytest.raises(RetriesExhaustedError):
            await call_with_retries(operation, max_attempts=4, base_delay=0.5, sleep=fake_sleep)

        assert sleep_calls == [0.5, 1.0, 2.0]

    async def test_permanent_errors_are_not_retried(
        self, fake_sleep: Any, sleep_calls: list[float]
    ) -> None:
        attempts = 0

        async def operation() -> str:
            nonlocal attempts
            attempts += 1
            raise LLMPermanentError("bad key")

        with pytest.raises(LLMPermanentError):
            await call_with_retries(operation, max_attempts=5, sleep=fake_sleep)

        assert attempts == 1
        assert sleep_calls == []

    async def test_timeouts_count_as_transient(self, fake_sleep: Any) -> None:
        attempts = 0

        async def operation() -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.TimeoutException("timed out")
            return "ok"

        assert await call_with_retries(operation, max_attempts=2, sleep=fake_sleep) == "ok"

    async def test_rejects_an_invalid_attempt_count(self, fake_sleep: Any) -> None:
        async def operation() -> str:
            return "ok"

        with pytest.raises(ValueError, match="max_attempts"):
            await call_with_retries(operation, max_attempts=0, sleep=fake_sleep)


class TestOpenAICompatibleClient:
    async def test_returns_message_content(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "model": "test-model",
                    "choices": [{"message": {"content": "book it in"}}],
                },
            )

        client = openai_client(httpx.MockTransport(handler))
        response = await client.generate(system="TASK: maintenance_note", user="hi", max_tokens=42)

        # Note: these tests inject their own AsyncClient to control the
        # transport, which deliberately bypasses the provider's own header
        # construction - so auth headers are asserted separately below.
        assert response.text == "book it in"
        assert captured["body"]["max_tokens"] == 42
        assert captured["body"]["model"] == "test-model"
        assert captured["body"]["messages"][0]["role"] == "system"

    async def test_a_self_constructed_client_sends_the_api_key(self) -> None:
        """Auth headers are set when the provider builds its own client."""
        client = OpenAICompatibleClient(
            api_key="test-key", base_url="https://llm.example/v1", model="m"
        )
        try:
            headers = client._client.headers  # noqa: SLF001
            assert headers["Authorization"] == "Bearer test-key"
            assert headers["Content-Type"] == "application/json"
        finally:
            await client.aclose()

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    async def test_retryable_statuses_are_transient(self, status: int) -> None:
        client = openai_client(
            httpx.MockTransport(lambda request: httpx.Response(status, text="busy"))
        )
        with pytest.raises(LLMTransientError):
            await client.generate(system="s", user="u")

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    async def test_client_errors_are_permanent(self, status: int) -> None:
        client = openai_client(
            httpx.MockTransport(lambda request: httpx.Response(status, text="nope"))
        )
        with pytest.raises(LLMPermanentError):
            await client.generate(system="s", user="u")

    async def test_malformed_success_body_is_transient(self) -> None:
        client = openai_client(
            httpx.MockTransport(lambda request: httpx.Response(200, json={"unexpected": 1}))
        )
        with pytest.raises(LLMTransientError):
            await client.generate(system="s", user="u")

    async def test_null_content_becomes_an_empty_string(self) -> None:
        client = openai_client(
            httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"choices": [{"message": {"content": None}}]}
                )
            )
        )
        assert (await client.generate(system="s", user="u")).text == ""

    async def test_generate_json_parses_the_reply(self) -> None:
        client = openai_client(
            httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"choices": [{"message": {"content": '{"ok": true}'}}]}
                )
            )
        )
        assert await client.generate_json(system="s", user="u") == {"ok": True}

    async def test_aclose_is_safe_on_an_injected_client(self) -> None:
        client = openai_client(httpx.MockTransport(lambda request: httpx.Response(200, json={})))
        await client.aclose()  # must not raise


class TestHuggingFaceInferenceClient:
    def _client(self, transport: httpx.MockTransport) -> HuggingFaceInferenceClient:
        return HuggingFaceInferenceClient(
            api_key="hf-key",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            client=httpx.AsyncClient(
                transport=transport, base_url="https://api-inference.huggingface.co/models"
            ),
        )

    async def test_parses_a_list_response(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=[{"generated_text": "llama says book it in"}])

        response = await self._client(httpx.MockTransport(handler)).generate(system="s", user="u")

        assert response.text == "llama says book it in"
        assert "Meta-Llama-3-8B-Instruct" in str(captured["path"])
        # HF rejects temperature == 0.
        assert captured["body"]["parameters"]["temperature"] > 0

    async def test_model_loading_503_is_transient(self) -> None:
        client = self._client(
            httpx.MockTransport(lambda request: httpx.Response(503, text="loading"))
        )
        with pytest.raises(LLMTransientError):
            await client.generate(system="s", user="u")

    async def test_malformed_body_is_transient(self) -> None:
        client = self._client(
            httpx.MockTransport(lambda request: httpx.Response(200, json={"nope": 1}))
        )
        with pytest.raises(LLMTransientError):
            await client.generate(system="s", user="u")


class TestStubLLMClient:
    @pytest.mark.parametrize(
        ("band", "fragment"),
        [
            ("high", "before its"),
            ("medium", "next depot visit"),
            ("low", "No air pressure system action"),
        ],
    )
    async def test_the_note_matches_the_band_in_the_prompt(self, band: str, fragment: str) -> None:
        """The stub must not tell a low-risk vehicle to visit the workshop."""
        response = await StubLLMClient().generate(
            system="TASK: maintenance_note", user=f"Risk band: {band}"
        )
        assert fragment in response.text

    async def test_an_unrecognised_band_falls_back_to_medium(self) -> None:
        response = await StubLLMClient().generate(
            system="TASK: maintenance_note", user="Risk band: unknown"
        )
        assert "next depot visit" in response.text

    async def test_a_prompt_without_a_band_still_returns_a_note(self) -> None:
        response = await StubLLMClient().generate(system="s", user="no band here")
        assert len(response.text) > 20

    async def test_the_raw_payload_records_what_it_saw(self) -> None:
        response = await StubLLMClient().generate(
            system="s", user="Risk band: high\nVehicle: TRK-1"
        )
        assert response.raw["stub"] is True
        assert response.raw["band"] == "high"


class TestFactory:
    def test_stub_needs_no_api_key(self) -> None:
        client = build_llm_client(Settings(llm_provider="stub", llm_api_key=None))
        assert isinstance(client, StubLLMClient)
        assert client.model == "stub-notes-v1"

    def test_openai_is_selected_by_configuration_alone(self) -> None:
        client = build_llm_client(
            Settings(llm_provider="openai", llm_api_key="sk-test", llm_model="gpt-4o-mini")
        )
        assert isinstance(client, OpenAICompatibleClient)
        assert client.model == "gpt-4o-mini"

    def test_huggingface_is_selected_by_configuration_alone(self) -> None:
        client = build_llm_client(
            Settings(llm_provider="huggingface", llm_api_key="hf-test", llm_model="llama")
        )
        assert isinstance(client, HuggingFaceInferenceClient)

    @pytest.mark.parametrize("provider", ["openai", "huggingface"])
    def test_a_networked_provider_without_a_key_fails_fast(self, provider: str) -> None:
        with pytest.raises(LLMPermanentError, match="LLM_API_KEY"):
            build_llm_client(Settings(llm_provider=provider, llm_api_key=None))
