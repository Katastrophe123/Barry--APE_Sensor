"""Turns a numeric prediction into a plain-English maintenance note.

This is the one place an LLM appears, and it is deliberately peripheral: the
prediction is the product, the note is a convenience for the operator reading
it. Consequently the service **always returns a note** - if the model is slow,
unauthorised or down, a deterministic template is used instead and the response
says so via ``generated_by``. A text generator being unavailable must never
turn into a failed prediction lookup.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.llm.base import LLMClient, LLMError
from app.llm.retry import SleepFn, call_with_retries
from app.logging_config import get_logger
from app.models.enums import RiskBand

logger = get_logger(__name__)

SYSTEM_PROMPT = """TASK: maintenance_note
You are a fleet maintenance advisor writing for a workshop technician.

Given a predictive-maintenance score for a truck's air pressure system (APS),
write a short, practical note: what to do, how urgently, and what to check.

Rules:
- 2 to 4 sentences, plain English, no bullet points, no headings.
- Say what action to take and by when; do not hedge.
- Reference only the sensor readings you are given. Never invent readings,
  part numbers, costs or policies.
- A probability is not a certainty: say "likely" or "elevated risk", never
  "the truck will fail".
- Do not explain what a probability is.
Output the note text only."""


@dataclass(frozen=True, slots=True)
class MaintenanceNote:
    """A note plus how it was produced."""

    text: str
    generated_by: str  # "llm" | "template"
    provider: str
    model: str


class MaintenanceNoteService:
    """Generates operator-facing notes, with a deterministic fallback."""

    MAX_SIGNALS = 10

    def __init__(
        self,
        *,
        llm: LLMClient,
        max_attempts: int = 2,
        backoff_base_seconds: float = 0.5,
        sleep: SleepFn | None = None,
    ) -> None:
        self._llm = llm
        self._max_attempts = max_attempts
        self._backoff_base_seconds = backoff_base_seconds
        self._sleep = sleep

    # --- prompt -----------------------------------------------------------
    def build_prompt(
        self,
        *,
        failure_probability: float,
        risk_band: RiskBand,
        vehicle_id: str | None,
        top_signals: list[str],
        model_version: str | None,
    ) -> str:
        signals = top_signals[: self.MAX_SIGNALS]
        lines = [
            f"Vehicle: {vehicle_id or 'unidentified'}",
            f"Predicted APS failure probability: {failure_probability:.1%}",
            f"Risk band: {risk_band.value}",
        ]
        if model_version:
            lines.append(f"Model version: {model_version}")
        if signals:
            lines.append("Notable sensor readings:")
            lines.extend(f"- {signal}" for signal in signals)
        else:
            lines.append("No individual sensor readings were supplied.")
        return "\n".join(lines)

    # --- fallback ---------------------------------------------------------
    @staticmethod
    def template_note(
        *,
        failure_probability: float,
        risk_band: RiskBand,
        vehicle_id: str | None,
        top_signals: list[str],
    ) -> str:
        """Deterministic note, used when the LLM is unavailable.

        Deliberately as useful as the generated version for the decision that
        matters - when to book the truck in - so a provider outage degrades
        wording, not operations.
        """
        vehicle = vehicle_id or "this vehicle"
        urgency = {
            RiskBand.HIGH: (
                "Book an APS inspection before the next long run; this is the "
                "highest-priority band."
            ),
            RiskBand.MEDIUM: "Schedule an APS inspection at the next depot visit.",
            RiskBand.LOW: "No APS action is required beyond the normal service schedule.",
        }[risk_band]
        parts = [
            f"{vehicle} has an estimated APS failure probability of "
            f"{failure_probability:.1%} ({risk_band.value} risk).",
            urgency,
        ]
        if risk_band is not RiskBand.LOW:
            parts.append(
                "Check compressor output and the system leak-down rate, and confirm "
                "the air dryer is cycling."
            )
        if top_signals:
            parts.append("Readings worth reviewing: " + ", ".join(top_signals[:5]) + ".")
        return " ".join(parts)

    # --- API --------------------------------------------------------------
    async def generate(
        self,
        *,
        failure_probability: float,
        risk_band: RiskBand,
        vehicle_id: str | None = None,
        top_signals: list[str] | None = None,
        model_version: str | None = None,
    ) -> MaintenanceNote:
        """Return a note, preferring the LLM but never failing."""
        signals = list(top_signals or [])
        fallback = self.template_note(
            failure_probability=failure_probability,
            risk_band=risk_band,
            vehicle_id=vehicle_id,
            top_signals=signals,
        )

        prompt = self.build_prompt(
            failure_probability=failure_probability,
            risk_band=risk_band,
            vehicle_id=vehicle_id,
            top_signals=signals,
            model_version=model_version,
        )

        async def operation() -> str:
            response = await self._llm.generate(
                system=SYSTEM_PROMPT, user=prompt, temperature=0.3, max_tokens=300
            )
            text = response.text.strip()
            if len(text) < 20:
                raise LLMError("Maintenance note was too short to be usable")
            return text

        try:
            text = await call_with_retries(
                operation,
                max_attempts=self._max_attempts,
                base_delay=self._backoff_base_seconds,
                operation_name="maintenance_note",
                sleep=self._sleep,
            )
        except LLMError as exc:
            logger.warning(
                "maintenance_note_fallback",
                reason=str(exc)[:200],
                provider=self._llm.name,
                risk_band=risk_band.value,
            )
            return MaintenanceNote(
                text=fallback,
                generated_by="template",
                provider=self._llm.name,
                model=self._llm.model,
            )

        logger.info(
            "maintenance_note_generated",
            provider=self._llm.name,
            model=self._llm.model,
            chars=len(text),
            risk_band=risk_band.value,
        )
        return MaintenanceNote(
            text=text,
            generated_by="llm",
            provider=self._llm.name,
            model=self._llm.model,
        )

    @staticmethod
    def top_signals_from_features(
        features: dict[str, float | None], *, limit: int = 5
    ) -> list[str]:
        """Pick the largest-magnitude readings as a crude "notable" set.

        Not a feature-importance explanation - it is magnitude only, and the
        README says so. SHAP would be the real answer; see "Future work".
        """
        numeric = [(name, float(value)) for name, value in features.items() if value is not None]
        numeric.sort(key=lambda pair: abs(pair[1]), reverse=True)
        return [f"{name} = {value:g}" for name, value in numeric[:limit]]
