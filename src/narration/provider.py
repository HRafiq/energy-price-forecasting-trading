"""Who writes the briefing, and the rule it is held to.

The desk briefing is three to five sentences of operator language about one tab.
A language model writes it, but the numbers are not its to invent: it receives the
payload (:mod:`src.narration.payload`) and may use nothing else, and every figure
it writes is checked afterwards (:mod:`src.narration.grounding`). A briefing whose
numbers fail that check is never shown.

Two providers implement the same interface:

* :class:`OpenAIProvider` calls the OpenAI API with ``OPENAI_API_KEY`` from the
  gitignored ``.env``. Without a key it is not selected, so the repository runs
  and its tests pass with no key and no network.
* :class:`TemplateProvider` writes the same brief from the payload with fixed
  sentences. It is what answers when no key is set, what the tests use, and the
  fallback when a call fails or comes back ungrounded.

``build_provider`` picks between them, so nothing above this module knows which
one answered; the response says so instead.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.config import REPO_ROOT

__all__ = [
    "BRIEFING_RULES",
    "FOLLOW_UPS",
    "Briefing",
    "NarrationError",
    "OpenAIProvider",
    "Provider",
    "TemplateProvider",
    "build_provider",
    "load_api_key",
]

#: The environment variable holding the key. Put it in the gitignored .env as
#: OPENAI_API_KEY=sk-...; nothing in the repository ever stores a key.
API_KEY_ENV = "OPENAI_API_KEY"
#: Small and quick: a briefing is a few sentences over a payload of a few hundred
#: numbers. Override with NARRATION_MODEL.
DEFAULT_MODEL = "gpt-4o-mini"
MODEL_ENV = "NARRATION_MODEL"
#: The canned questions under each briefing.
FOLLOW_UPS = {
    "why_this_dispatch": "Why this dispatch?",
    "what_changed": "What changed vs yesterday?",
    "explain_the_miss": "Explain the miss",
}
#: The rules the prompt states, and the grounding check enforces afterwards.
BRIEFING_RULES = """You are a power trading desk analyst writing a short briefing
for the operator of a 1 MW battery in the German day-ahead market.

Rules, in order of importance:
1. Every number you write must appear in the payload. Do not add, derive,
   average, convert or estimate any figure. If a number is not in the payload,
   do not use it.
2. Three to five sentences. No lists, no headings, no markdown.
3. Plain operator language: what happened, and what it cost or earned.
4. Say what the numbers show, not what should be done about it.
5. If the payload says prices are not published yet, say the day is not settled
   rather than guessing what it earned."""


class NarrationError(RuntimeError):
    """The briefing could not be written."""


@dataclass(frozen=True)
class Briefing:
    """One briefing and where its words came from."""

    text: str
    provider: str
    model: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"text": self.text, "provider": self.provider, "model": self.model}


class Provider(Protocol):
    """Anything that can write a briefing from a payload."""

    name: str

    def write(self, payload: dict[str, Any], question: str | None = None) -> Briefing:
        """Three to five sentences about ``payload``, answering ``question``."""


def load_api_key(env_path: Path | None = None) -> str | None:
    """The OpenAI key from the environment, or from the gitignored .env.

    Never returns the key to a caller that logs it: the provider holds it and the
    response only ever says which provider answered.
    """
    key = os.environ.get(API_KEY_ENV)
    if key:
        return key.strip() or None
    path = env_path or REPO_ROOT / ".env"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition("=")
        if name.strip() == API_KEY_ENV:
            return value.strip().strip('"').strip("'") or None
    return None


def _sentence(value: Any, unit: str = "", places: int = 2) -> str:
    """A number as the briefing writes it, so the grounding check can match it.

    A figure keeps its decimals unless it is exactly whole: a capture ratio of
    90.09% is written 90.09%, not 90%, which is what a desk analyst would say and
    what the payload can support.
    """
    if isinstance(value, float):
        if value == int(value):
            return f"{int(value):,}{unit}"
        return f"{value:,.{places}f}{unit}".rstrip("0").rstrip(".")
    return f"{value}{unit}"


@dataclass
class TemplateProvider:
    """Writes the briefing from the payload with fixed sentences.

    It states only what it reads, so it is grounded by construction. It answers
    when no key is set, and stands in for the model in every test.
    """

    name: str = "template"

    def write(self, payload: dict[str, Any], question: str | None = None) -> Briefing:
        tab = payload.get("tab", "overview")
        writer = {
            "overview": self._overview,
            "forecast": self._forecast,
            "trading": self._trading,
            "model_health": self._model_health,
        }.get(str(tab))
        if writer is None:
            raise NarrationError(f"no briefing for tab {tab!r}")
        text = writer(payload)
        if question:
            text = f"{text} {self._answer(payload, question)}"
        return Briefing(text, self.name)

    def _answer(self, payload: dict[str, Any], question: str) -> str:
        """The follow-up, answered from the payload or declined for want of it."""
        asked = question.lower().rstrip("?")
        previous = payload.get("previous_day")
        if "changed" in asked and previous:
            change = float(previous["pnl_change_eur"])
            direction = "more" if change >= 0 else "less"
            return (
                f"Asked {asked}: on {previous['day']} it earned "
                f"{_sentence(previous['pnl_eur'])} EUR, so this day is "
                f"{_sentence(previous['pnl_change_size_eur'])} EUR {direction}."
            )
        return f"Asked {asked}: the payload holds only the figures above."

    def _overview(self, payload: dict[str, Any]) -> str:
        fan, dispatch = payload["forecast"], payload["dispatch"]
        day = payload["day"]
        if not fan.get("prices_published", False):
            return (
                f"The schedule for {day} was committed before the "
                f"{payload['gate_local']} gate, from the forecast issued at "
                f"{payload['issued_local']}. "
                "Prices for the day are not published yet, so it is not settled."
            )
        low = _sentence(fan["lowest_price_eur_mwh"])
        high = _sentence(fan["highest_price_eur_mwh"])
        return (
            f"On {day} the price ran from {low} EUR/MWh at "
            f"{fan['lowest_price_at']} to {high} at "
            f"{fan['highest_price_at']}. The battery charged over "
            f"{dispatch['charging_periods']} periods and discharged over "
            f"{dispatch['discharging_periods']}, earning "
            f"{_sentence(dispatch['pnl_eur'])} EUR against "
            f"{_sentence(dispatch['perfect_foresight_pnl_eur'])} with perfect "
            "foresight, "
            f"a shortfall of {_sentence(dispatch['shortfall_eur'])} EUR."
        )

    def _forecast(self, payload: dict[str, Any]) -> str:
        window, worst = payload["window"], payload.get("worst_hour")
        intervals = ", ".join(
            f"{_sentence(i['nominal'] * 100)}% band held "
            f"{_sentence(i['coverage'] * 100)}%"
            for i in payload["intervals"]
        )
        hour = (
            f" The median was furthest out at hour {worst['hour']}, "
            f"{_sentence(worst['mae_eur_mwh'])} EUR/MWh on average."
            if worst
            else ""
        )
        return (
            f"Over {window['traded_days']} traded days from {window['first_day']} to "
            f"{window['last_day']}, {intervals}.{hour}"
        )

    def _trading(self, payload: dict[str, Any]) -> str:
        """An empty window leaves the API's KPIs null; say less rather than wrong."""
        kpis, window = payload["kpis"], payload["window"]
        ratio, ceiling = (
            kpis.get("capture_ratio"),
            kpis.get("perfect_foresight_pnl_eur"),
        )
        against = (
            f", {_sentence(ratio * 100)}% of the {_sentence(ceiling)} EUR perfect "
            "foresight made"
            if ratio is not None and ceiling is not None
            else ""
        )
        turns = kpis.get("cycles_per_day")
        cycles = (
            ""
            if turns is None
            else ", cycling once a day"
            if float(turns) == 1.0
            else f", cycling {_sentence(turns)} times a day"
        )
        drawdown = payload["max_drawdown_eur"].get("selected")
        deepest = (
            f" The deepest fall from a peak was {_sentence(drawdown)} EUR."
            if drawdown is not None
            else ""
        )
        return (
            f"Across {window['traded_days']} traded days the battery earned "
            f"{_sentence(kpis['pnl_eur'])} EUR{against}{cycles}.{deepest}"
        )

    def _model_health(self, payload: dict[str, Any]) -> str:
        drift = payload["ops"]["drift"]
        counts = payload["incident_counts"].get("type", {})
        by_type = ", ".join(
            f"{count} {kind.replace('_', ' ')}"
            for kind, count in sorted(counts.items())
            if count
        )
        coverage, ratio = drift["coverage"], drift["pinball_ratio"]
        return (
            f"Rolling coverage stands at {_sentence(coverage['value'] * 100)}% "
            f"against an alert line of {_sentence(coverage['threshold'] * 100)}%, "
            f"and the pinball ratio at {_sentence(ratio['value'])} against "
            f"{_sentence(ratio['threshold'])}. The incident log holds {by_type}."
        )


@dataclass
class OpenAIProvider:
    """Calls the OpenAI API, and refuses to answer without a key."""

    api_key: str
    model: str = DEFAULT_MODEL
    name: str = "openai"
    timeout_s: float = 30.0

    def write(self, payload: dict[str, Any], question: str | None = None) -> Briefing:
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on the extra
            raise NarrationError(
                "the openai package is not installed; add it with "
                "`uv add openai` or leave OPENAI_API_KEY unset to use the template"
            ) from exc
        client = OpenAI(api_key=self.api_key, timeout=self.timeout_s)
        ask = question or "Write the briefing for this tab."
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": BRIEFING_RULES},
                {
                    "role": "user",
                    "content": f"{ask}\n\nPayload:\n{json.dumps(payload, indent=2)}",
                },
            ],
            temperature=0.2,
            max_tokens=320,
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise NarrationError("the model returned an empty briefing")
        return Briefing(text, self.name, self.model)


def build_provider(env_path: Path | None = None) -> Provider:
    """The model when a key is configured, the template writer otherwise."""
    key = load_api_key(env_path)
    if not key:
        return TemplateProvider()
    return OpenAIProvider(api_key=key, model=os.environ.get(MODEL_ENV, DEFAULT_MODEL))
