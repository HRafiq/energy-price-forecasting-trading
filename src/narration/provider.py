"""Who writes the briefing, and the rule it is held to.

The desk briefing is three to five sentences of operator language about one tab.
By default a deterministic writer produces it from the page's own numbers. A
language model can be switched on to write it instead, but the numbers are not its
to invent: it receives the payload (:mod:`src.narration.payload`) and may use nothing
else, and every figure it writes is checked afterwards
(:mod:`src.narration.grounding`). A briefing whose numbers fail that check is never
shown. The model gets one second draft, told which figures failed (:class:`Retry`);
if that fails too, the template answers.

The model is off by default because the check proves a figure is on the page, not
that a sentence gives it the right meaning, and measured against the real model that
is the gap that shows (production notes, M5: a grounded briefing that misreads the
page).

Three providers implement the same interface:

* :class:`TemplateProvider` writes the brief from the payload with fixed sentences.
  It is grounded by construction, answers unless a model is switched on, is what
  the tests use, and is the fallback when a call fails or comes back ungrounded.
* :class:`OpenAIProvider` calls the OpenAI API with ``OPENAI_API_KEY``, and
  :class:`AnthropicProvider` the Anthropic API with ``ANTHROPIC_API_KEY``. Either
  answers only when ``NARRATION_PROVIDER`` names it and its key is set, both read
  from the environment or the gitignored ``.env`` (see ``.env.example``). A failed
  call, a wrong key or a rate limit, becomes a ``NarrationError``, so the caller
  falls back rather than failing.

``build_provider`` picks between them, so nothing above this module knows which
one answered; the response says so instead.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.config import REPO_ROOT

logger = logging.getLogger(__name__)

__all__ = [
    "BRIEFING_RULES",
    "FOLLOW_UPS",
    "MAX_ATTEMPTS",
    "AnthropicProvider",
    "Briefing",
    "NarrationError",
    "OpenAIProvider",
    "Provider",
    "Retry",
    "TemplateProvider",
    "build_provider",
    "env_value",
    "load_api_key",
]

#: The environment variables holding the keys, set in the gitignored .env (see
#: .env.example). Nothing in the repository ever stores a key.
API_KEY_ENV = "OPENAI_API_KEY"
ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
#: Switches a model on: "openai" or "anthropic". A key alone does not.
PROVIDER_ENV = "NARRATION_PROVIDER"
#: A briefing is a few sentences over a payload of a few hundred numbers, so both
#: defaults are the providers' small, cheap models. Override with NARRATION_MODEL.
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
MODEL_ENV = "NARRATION_MODEL"
MAX_TOKENS = 320
#: One network retry per call, 15 seconds each, and two drafts at most: a briefing
#: that has not arrived in about a minute is replaced by the template rather than
#: holding the page. A briefing normally arrives in 2 to 5 seconds.
MAX_RETRIES = 1
TIMEOUT_S = 15.0
#: Attempts the model gets at one briefing: the first draft, and one rewrite told
#: which figures failed the check. A third would rarely fix what two did not, and
#: would make the operator wait for a template anyway.
MAX_ATTEMPTS = 2
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
   average, convert or estimate any figure: no differences, sums, ratios or
   percentages of your own. If a number is not in the payload, do not use it.
2. Write every figure the way the payload holds it:
   - a date exactly as the payload writes it, such as 2026-09-14, never with a
     month name;
   - a clock time as HH:MM, such as 19:45;
   - a negative figure with its minus sign, such as -3.39, never as "a negative
     3.39" or "a loss of 3.39";
   - a share such as 0.8957 may be written as 89.57%, and any figure may be
     rounded to two decimals.
3. Name an hour block such as "18-20" in words, such as "the evening hours",
   never by its digits.
4. Three to five sentences. No lists, no headings, no markdown.
5. Plain operator language: what happened, and what it cost or earned.
6. Say what the numbers show, not what should be done about it.
7. Only when the payload holds "prices_published": false, say the day is not
   settled rather than guessing what it earned. Otherwise do not mention
   settlement or published prices."""


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


@dataclass(frozen=True)
class Retry:
    """A draft the grounding check refused, and the figures it refused."""

    draft: str
    rejected: tuple[str, ...]


class Provider(Protocol):
    """Anything that can write a briefing from a payload."""

    name: str

    def write(
        self,
        payload: dict[str, Any],
        question: str | None = None,
        retry: Retry | None = None,
    ) -> Briefing:
        """Three to five sentences about ``payload``, answering ``question``.

        With ``retry``, the same briefing again, without the refused figures.
        """


def env_value(name: str, env_path: Path | None = None) -> str | None:
    """A setting from the environment, else from the gitignored .env; None if empty."""
    value = os.environ.get(name)
    if value is not None and value.strip():
        return value.strip()
    path = env_path or REPO_ROOT / ".env"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, raw = line.partition("=")
        if key.strip() == name:
            return raw.strip().strip('"').strip("'") or None
    return None


def load_api_key(env_path: Path | None = None, name: str = API_KEY_ENV) -> str | None:
    """A provider key from the environment, or from the gitignored .env.

    Never returns the key to a caller that logs it: the provider holds it and the
    response only ever says which provider answered.
    """
    return env_value(name, env_path)


def _user_message(payload: dict[str, Any], question: str | None) -> str:
    ask = question or "Write the briefing for this tab."
    return f"{ask}\n\nPayload:\n{json.dumps(payload, indent=2)}"


def _retry_message(retry: Retry) -> str:
    figures = ", ".join(f'"{figure}"' for figure in retry.rejected)
    return (
        f"These figures in your briefing are not in the payload: {figures}. Write "
        "the briefing again without them. Quote dates and times exactly as the "
        "payload writes them, keep minus signs, name hour blocks in words, and "
        "state no figure you worked out yourself."
    )


def _conversation(
    payload: dict[str, Any], question: str | None, retry: Retry | None
) -> list[Any]:
    """The user turns, with the refused draft and why it was refused on a retry.

    Both SDKs take this shape of turn; each types it as its own message class.
    """
    turns: list[Any] = [{"role": "user", "content": _user_message(payload, question)}]
    if retry is not None:
        turns += [
            {"role": "assistant", "content": retry.draft},
            {"role": "user", "content": _retry_message(retry)},
        ]
    return turns


def _call_failed(provider: str, exc: Exception) -> str:
    """Why a call failed, without anything the exception carries about the request."""
    status = getattr(exc, "status_code", None)
    detail = f" (HTTP {status})" if isinstance(status, int) else ""
    return f"the {provider} call failed: {type(exc).__name__}{detail}"


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
    unless a model is switched on, and stands in for the model in every test.
    """

    name: str = "template"

    def write(
        self,
        payload: dict[str, Any],
        question: str | None = None,
        retry: Retry | None = None,
    ) -> Briefing:
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
    timeout_s: float = TIMEOUT_S

    def write(
        self,
        payload: dict[str, Any],
        question: str | None = None,
        retry: Retry | None = None,
    ) -> Briefing:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on the install
            raise NarrationError(
                "the openai package is not installed; run `uv sync` or leave "
                "NARRATION_PROVIDER unset to use the template"
            ) from exc
        try:
            client = OpenAI(
                api_key=self.api_key, timeout=self.timeout_s, max_retries=MAX_RETRIES
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": BRIEFING_RULES},
                    *_conversation(payload, question, retry),
                ],
                temperature=0.2,
                max_tokens=MAX_TOKENS,
            )
        except Exception as exc:
            raise NarrationError(_call_failed(self.name, exc)) from exc
        choices = response.choices or []
        text = (choices[0].message.content or "").strip() if choices else ""
        if not text:
            raise NarrationError("the model returned an empty briefing")
        return Briefing(text, self.name, self.model)


@dataclass
class AnthropicProvider:
    """Calls the Anthropic API, and refuses to answer without a key."""

    api_key: str
    model: str = DEFAULT_ANTHROPIC_MODEL
    name: str = "anthropic"
    timeout_s: float = TIMEOUT_S

    def write(
        self,
        payload: dict[str, Any],
        question: str | None = None,
        retry: Retry | None = None,
    ) -> Briefing:
        try:
            from anthropic import Anthropic
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on the install
            raise NarrationError(
                "the anthropic package is not installed; run `uv sync` or leave "
                "NARRATION_PROVIDER unset to use the template"
            ) from exc
        try:
            client = Anthropic(
                api_key=self.api_key, timeout=self.timeout_s, max_retries=MAX_RETRIES
            )
            response = client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=BRIEFING_RULES,
                messages=_conversation(payload, question, retry),
            )
        except Exception as exc:
            raise NarrationError(_call_failed(self.name, exc)) from exc
        text = "".join(getattr(block, "text", "") for block in response.content).strip()
        if not text:
            raise NarrationError("the model returned an empty briefing")
        return Briefing(text, self.name, self.model)


def build_provider(env_path: Path | None = None) -> Provider:
    """The deterministic writer, unless a model is switched on.

    A model writes the briefing only when ``NARRATION_PROVIDER`` names it, "openai"
    or "anthropic", and that provider's key is set. A key on its own does not switch
    a model on, and neither does a named provider whose key is missing or a name
    that is neither. ``NARRATION_MODEL`` overrides the named provider's default
    model. Each setting is read from the environment first, then from .env.
    """
    choice = (env_value(PROVIDER_ENV, env_path) or "").lower()
    model = env_value(MODEL_ENV, env_path)
    if choice and choice not in ("openai", "anthropic", "template"):
        logger.warning(
            "%s=%r is not openai or anthropic; the template writes",
            PROVIDER_ENV,
            choice,
        )
    if choice == "openai" and (openai_key := env_value(API_KEY_ENV, env_path)):
        return OpenAIProvider(api_key=openai_key, model=model or DEFAULT_MODEL)
    if choice == "anthropic" and (
        anthropic_key := env_value(ANTHROPIC_KEY_ENV, env_path)
    ):
        return AnthropicProvider(
            api_key=anthropic_key, model=model or DEFAULT_ANTHROPIC_MODEL
        )
    if choice in ("openai", "anthropic"):
        logger.warning(
            "%s=%s but its key is not set; the template writes", PROVIDER_ENV, choice
        )
    return TemplateProvider()
