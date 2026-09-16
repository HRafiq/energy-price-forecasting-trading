"""Check that a briefing only states numbers the payload already contains.

The desk briefing is written by a language model, so the rule that makes it
trustworthy has to be enforced outside the model: every figure in the prose must
appear in the payload the model was given (:mod:`src.narration.payload`). This
module reads the numbers back out of the prose and looks for each one.

Matching allows for the ways a number is written rather than what it means:

* thousands separators and currency symbols, so "€1,234.56" is 1234.56;
* rounding, half up or half to even, so 90.06 supports "90.1" but not "91";
* percentages, so a payload's 0.9006 supports "90.1%";
* a minus sign written as a hyphen.

Rounding is the only latitude. A count of 19 does not support "20" and 964.46
does not support "965": a briefing that is one off is wrong in the way that
matters, because it arrives beside true figures and borrows their authority. A
percentage must come from a share: the payload holding 29 traded days does not
support "29% of perfect foresight".

Two kinds of text are handled specially. A clock time or a date the payload holds
may be quoted as written, so "19:45" is blanked out of the prose before it is read
for numbers; only a whole token is blanked, so the "00:00" inside "200:000" is
left alone and the figure around it is still checked. And a figure spelled out in
words, or written in numerals this module cannot value, is treated as unsupported
rather than ignored, because it cannot be looked up.

It cannot tell whether a sentence draws the right conclusion from a number, and it
checks the prose against the payload, not the payload against the page.

    report = check_grounding(prose, payload)
    if not report.grounded:
        ...  # report.unsupported lists what was not found
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from src.narration.payload import payload_labels, payload_numbers

__all__ = [
    "Claim",
    "GroundingReport",
    "check_grounding",
    "numbers_in_prose",
    "strip_numerals",
]

#: A number as a person writes it: 1,234.56 or 90.1% or -12.
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?\s*%?")
#: The only labels a briefing may quote as written: a clock time or an ISO date.
#: A block name such as "18-20" is not one of them, so its digits must be numbers
#: the data holds.
_QUOTABLE = re.compile(r"\d{1,2}:\d{2}|\d{4}-\d{2}-\d{2}")
#: A figure written in words. The briefing is told to use numerals, so a magnitude
#: word or a hyphenated compound is a claim this module cannot value, and what it
#: cannot value it cannot support.
_SPELLED = re.compile(
    r"\b(?:hundred|thousand|million|billion)\b"
    r"|\b(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[- ]"
    r"(?:one|two|three|four|five|six|seven|eight|nine)\b",
    re.IGNORECASE,
)
#: Every word that names a number, for taking figures out of a typed question.
_NUMBER_WORD = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|"
    r"billion)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Claim:
    """One number as the briefing wrote it.

    A claim this module cannot value, such as a figure spelled out in words,
    carries a value of NaN and can never be supported.
    """

    text: str
    value: float
    percent: bool

    @property
    def decimals(self) -> int:
        """How precisely it was written, which sets how precisely to match it."""
        _, _, fraction = self.text.replace("%", "").strip().partition(".")
        return len(fraction)


@dataclass(frozen=True)
class GroundingReport:
    """What the check found."""

    claims: tuple[Claim, ...]
    unsupported: tuple[Claim, ...]

    @property
    def grounded(self) -> bool:
        return not self.unsupported

    def as_dict(self) -> dict[str, Any]:
        return {
            "grounded": self.grounded,
            "claims": len(self.claims),
            "unsupported": [claim.text for claim in self.unsupported],
        }


def numbers_in_prose(prose: str) -> tuple[Claim, ...]:
    """Every number the briefing states, in the order it states them."""
    claims = []
    for match in _NUMBER.finditer(prose):
        raw = match.group(0).strip().rstrip(",.")
        percent = raw.endswith("%")
        digits = raw.rstrip("%").strip().replace(",", "")
        try:
            value = float(digits)
        except ValueError:  # pragma: no cover - the pattern guarantees a number
            continue
        claims.append(Claim(raw, value, percent))
    return tuple(claims)


def _unvaluable(prose: str) -> tuple[Claim, ...]:
    """Figures the check cannot look up: spelled out, or written in odd numerals.

    A briefing saying "nine hundred and forty euros" is stating a figure as surely
    as one saying 940, and a half sign or a Roman numeral is a digit this module
    does not read. Neither can be found in a payload, so both are unsupported.
    """
    found = [
        Claim(match.group(0), math.nan, False) for match in _SPELLED.finditer(prose)
    ]
    odd = {
        character
        for character in prose
        if not character.isascii() and unicodedata.numeric(character, None) is not None
    }
    found.extend(Claim(character, math.nan, False) for character in sorted(odd))
    return tuple(found)


def strip_numerals(text: str) -> str:
    """The text with every figure removed, whether written in digits or in words.

    Used on a question before it is echoed back inside an answer: a figure typed
    into the question would otherwise read as one of the day's own.
    """
    without_words = _NUMBER_WORD.sub("", text)
    without_digits = "".join(
        ""
        if unicodedata.numeric(character, None) is not None and not character.isalpha()
        else character
        for character in without_words
    )
    return re.sub(r"\s{2,}", " ", without_digits).strip(" -,")


def _quotable_labels(labels: set[str]) -> set[str]:
    """The clock times and dates in a payload, which a briefing may quote.

    A compound label contributes its parts, so a payload holding
    "2026-09-13 11:40" also allows a sentence to say "11:40".
    """
    return {match.group(0) for label in labels for match in _QUOTABLE.finditer(label)}


def _mask_labels(prose: str, labels: set[str]) -> str:
    """Blank out quoted labels, so the digits inside a time are not read as claims.

    Only a whole token is blanked. "00:00" inside "200:000" is part of a longer
    figure, so it is left alone and the figure is checked like any other.
    """
    masked = prose
    for label in sorted(labels, key=len, reverse=True):
        masked = re.sub(rf"(?<![\d:-]){re.escape(label)}(?![\d:-])", " ", masked)
    return masked


def _half_up(value: float, places: int) -> Decimal | None:
    """The value rounded half away from zero, as a person rounds by hand."""
    try:
        return Decimal(repr(value)).quantize(
            Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP
        )
    except (InvalidOperation, ValueError):  # pragma: no cover - guarded by caller
        return None


def _same(option: float, claim: Claim) -> bool:
    """Equal to the claim once both are rounded as the claim was written."""
    places = claim.decimals
    if round(option, places) == round(claim.value, places):
        return True
    # Python rounds a half to even, so 401.25 gives 401.2; a briefing writing
    # 401.3 has rounded the same number the other way and is not wrong.
    rounded, written = _half_up(option, places), _half_up(claim.value, places)
    return rounded is not None and rounded == written


def _supports(claim: Claim, candidate: float) -> bool:
    """Does a payload number support the claim as it was written?"""
    if not claim.percent:
        return _same(candidate, claim)
    # A percentage comes from a share. Without this, every number in the payload
    # would license itself as a percentage: 29 traded days would support "29% of
    # perfect foresight", and 19 charging periods "1900%".
    if not -1.0 <= candidate <= 1.0:
        return False
    return _same(candidate * 100.0, claim)


def check_grounding(prose: str, payload: Any) -> GroundingReport:
    """Every number in ``prose`` must appear in ``payload``."""
    available = payload_numbers(payload)
    labels = _quotable_labels(payload_labels(payload))
    claims = numbers_in_prose(_mask_labels(prose, labels)) + _unvaluable(prose)
    unsupported = tuple(
        claim
        for claim in claims
        if not math.isfinite(claim.value)
        or not any(_supports(claim, candidate) for candidate in available)
    )
    return GroundingReport(claims, unsupported)
