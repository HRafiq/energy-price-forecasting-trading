"""The desk briefing: what it may say, how that is checked, and who writes it."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest

from src.narration import grounding
from src.narration import payload as payload_mod
from src.narration import provider as provider_mod

OVERVIEW: dict[str, Any] = {
    "tab": "overview",
    "run_id": "backtest-2026-09-14",
    "day": "2026-09-14",
    "issued_local": "2026-09-13 11:40",
    "gate_local": "2026-09-13 12:00",
    "forecast": {
        "periods": 96,
        "prices_published": True,
        "highest_price_eur_mwh": 740.01,
        "highest_price_at": "19:45",
        "lowest_price_eur_mwh": 152.0,
        "lowest_price_at": "16:00",
        "widest_band_eur_mwh": 401.25,
    },
    "dispatch": {
        "pnl_eur": 964.46,
        "perfect_foresight_pnl_eur": 1034.09,
        "shortfall_eur": 69.63,
        "charging_periods": 19,
        "discharging_periods": 16,
        "first_charge_at": "02:00",
        "last_discharge_at": "20:30",
        "cheapest_charge_price_eur_mwh": 152.0,
        "dearest_discharge_price_eur_mwh": 740.01,
    },
}
FORECAST: dict[str, Any] = {
    "tab": "forecast",
    "window": {"traded_days": 29, "first_day": "2026-08-16", "last_day": "2026-09-14"},
    "intervals": [
        {"nominal": 0.5, "coverage": 0.3168},
        {"nominal": 0.9, "coverage": 0.6742},
    ],
    "worst_hour": {"hour": 19, "mae_eur_mwh": 61.24},
    "asymmetry": {"gap_eur": 1019.32, "days": 29, "blocks": []},
}
TRADING: dict[str, Any] = {
    "tab": "trading",
    "window": {"traded_days": 29, "first_day": "2026-08-16", "last_day": "2026-09-14"},
    "battery": {"power_mw": 1.0, "duration_h": 2},
    "kpis": {
        "pnl_eur": 8755.24,
        "capture_ratio": 0.8957,
        "perfect_foresight_pnl_eur": 9774.56,
        "cycles_per_day": 1.71,
    },
    "max_drawdown_eur": {"selected": 128.4},
    "selected_strategy": "median",
}
MODEL_HEALTH: dict[str, Any] = {
    "tab": "model_health",
    "ops": {
        "drift": {
            "coverage": {"value": 0.6749, "threshold": 0.74},
            "pinball_ratio": {"value": 1.9372, "threshold": 1.5},
        }
    },
    "incident_counts": {"type": {"data_gap": 1, "drift": 9, "tail_miss": 11}},
    "recent_incidents": [],
}
PAYLOADS = {
    "overview": OVERVIEW,
    "forecast": FORECAST,
    "trading": TRADING,
    "model_health": MODEL_HEALTH,
}


class _StubRun:
    """Just enough of a run for the checks that refuse before reading data."""

    run_id = "backtest-2026-09-14"
    manifest: ClassVar[dict[str, str]] = {
        "model": "lightgbm_conformal",
        "run_kind": "backtest",
    }


def test_the_four_tabs_are_the_dashboard_tabs() -> None:
    assert payload_mod.TABS == ("overview", "forecast", "trading", "model_health")


def test_an_unknown_tab_is_refused_before_anything_is_read() -> None:
    with pytest.raises(payload_mod.PayloadError, match="unknown tab"):
        payload_mod.build_payload(
            cast(Any, None),
            None,
            tab="pnl",
            day=date(2026, 9, 14),
            window="last30",
            battery={},
        )


def test_model_health_needs_a_health_export() -> None:
    with pytest.raises(payload_mod.PayloadError, match="health export"):
        payload_mod.build_payload(
            cast(Any, _StubRun()),
            None,
            tab="model_health",
            day=date(2026, 9, 14),
            window="last30",
            battery={},
        )


def test_a_payloads_numbers_are_numbers_and_its_labels_are_labels() -> None:
    numbers = payload_mod.payload_numbers(OVERVIEW)
    labels = payload_mod.payload_labels(OVERVIEW)

    assert 964.46 in numbers and 1034.09 in numbers and 96.0 in numbers
    # The digits inside "19:45" are not numbers the briefing may quote: mining
    # them would let a clock time license a count of 45 that is nowhere in the
    # data. The label is matched whole instead.
    assert 45.0 not in numbers and "19:45" in labels
    assert "2026-09-14" in labels
    # True must not be read as 1.
    assert payload_mod.payload_numbers({"flag": True}) == set()


def test_a_figure_one_off_the_data_is_not_close_enough() -> None:
    # 19 periods and 964.46 EUR are in the payload; these are not.
    counted = grounding.check_grounding("It charged over 20 periods.", OVERVIEW)
    earned = grounding.check_grounding("It finished 965 EUR ahead.", OVERVIEW)

    assert not counted.grounded and [c.text for c in counted.unsupported] == ["20"]
    assert not earned.grounded and [c.text for c in earned.unsupported] == ["965"]


def test_a_date_cannot_license_a_figure_that_is_not_in_the_data() -> None:
    # "2026-09-14" and "2026-09-13" are labels in the payload; -14 and -13 are
    # not losses the battery made.
    invented = grounding.check_grounding(
        "The evening block gave back -14 EUR after a -13 EUR morning.", OVERVIEW
    )
    quoted = grounding.check_grounding(
        "The schedule for 2026-09-14 was committed before the 2026-09-13 12:00 gate.",
        OVERVIEW,
    )

    assert not invented.grounded
    assert [c.text for c in invented.unsupported] == ["-14", "-13"]
    assert quoted.grounded


def test_a_percentage_may_be_rounded_but_not_moved() -> None:
    # Capture is 0.8957, which a sentence may write 89.57%, 89.6% or 90%.
    for prose in ("Capture was 90%.", "Capture was 89.6%.", "Capture was 89.57%."):
        assert grounding.check_grounding(prose, TRADING).grounded, prose
    assert not grounding.check_grounding("Capture was 89%.", TRADING).grounded


def test_an_empty_window_leaves_the_kpis_null_and_the_briefing_still_writes() -> None:
    empty = TRADING | {
        "window": {"traded_days": 0, "first_day": None, "last_day": None},
        "kpis": {
            "pnl_eur": 0.0,
            "capture_ratio": None,
            "perfect_foresight_pnl_eur": 0.0,
            "cycles_per_day": None,
        },
        "max_drawdown_eur": {},
    }

    briefing = provider_mod.TemplateProvider().write(empty)

    assert "capture" not in briefing.text.lower() and "cycling" not in briefing.text
    assert grounding.check_grounding(briefing.text, empty).grounded


def test_prose_that_quotes_the_payload_is_grounded() -> None:
    prose = (
        "The battery earned 964.46 EUR against 1,034.09 with perfect foresight, "
        "a shortfall of 69.63 EUR after the price peaked at 740.01 at 19:45."
    )

    report = grounding.check_grounding(prose, OVERVIEW)

    assert report.grounded and report.unsupported == ()
    # Four claims, not six: "19:45" is a label the payload holds, so it is blanked
    # out before the prose is read and its digits are never claims at all.
    assert [claim.text for claim in report.claims] == [
        "964.46",
        "1,034.09",
        "69.63",
        "740.01",
    ]
    assert report.as_dict()["unsupported"] == []


def test_a_number_from_nowhere_is_caught() -> None:
    prose = "It earned 964.46 EUR, and the evening block alone brought 1,742 EUR."

    report = grounding.check_grounding(prose, OVERVIEW)

    assert not report.grounded
    assert [claim.text for claim in report.unsupported] == ["1,742"]


def test_a_percentage_matches_the_share_it_came_from() -> None:
    ok = grounding.check_grounding("Capture was 89.6% of perfect foresight.", TRADING)
    wrong = grounding.check_grounding(
        "Capture was 94.2% of perfect foresight.", TRADING
    )

    assert ok.grounded, [c.text for c in ok.unsupported]
    assert not wrong.grounded


def test_rounding_and_separators_are_allowed_but_a_wrong_figure_is_not() -> None:
    rounded = grounding.check_grounding("It earned 964 EUR.", OVERVIEW)
    separated = grounding.check_grounding(
        "Perfect foresight made 1,034.09 EUR.", OVERVIEW
    )
    wrong = grounding.check_grounding("It earned 9,644 EUR.", OVERVIEW)

    assert rounded.grounded and separated.grounded
    assert not wrong.grounded


def test_a_count_must_be_a_count_the_log_actually_holds() -> None:
    # The log holds 1 data gap and 9 drift alerts. A small number is not excused
    # for being small: nothing in this payload is, or rounds to, 4.
    real = grounding.check_grounding(
        "The day had 1 gap and 9 drift alerts.", MODEL_HEALTH
    )
    invented = grounding.check_grounding("The day had 4 gaps.", MODEL_HEALTH)

    assert real.grounded
    assert not invented.grounded


def test_a_time_may_be_quoted_but_not_hidden_inside_a_figure() -> None:
    quoted = grounding.check_grounding("The first charge was at 02:00.", OVERVIEW)
    # "02:00" is in there, but as part of a figure that is not in the payload.
    hidden = grounding.check_grounding("It earned 202:000 EUR overnight.", OVERVIEW)

    assert quoted.grounded
    assert not hidden.grounded


def test_a_block_name_is_not_a_figure_the_briefing_may_quote() -> None:
    payload = FORECAST | {
        "asymmetry": {
            "gap_eur": 1019.32,
            "days": 29,
            "blocks": [{"block": "18-20", "gap_eur": 512.0}],
        }
    }

    report = grounding.check_grounding("The gap widened by 18-20 EUR.", payload)

    assert not report.grounded


def test_a_percentage_must_come_from_a_share() -> None:
    # 29 traded days and 1.71 cycles a day are in the payload, but neither is a
    # share, so neither licenses itself as a percentage.
    for prose in ("Capture was 29%.", "Capture was 1.71%.", "It captured 1900%."):
        assert not grounding.check_grounding(prose, TRADING).grounded, prose


def test_a_figure_that_cannot_be_valued_cannot_be_supported() -> None:
    spelled = grounding.check_grounding(
        "The evening block alone brought in nine hundred and forty euros.", OVERVIEW
    )
    odd = grounding.check_grounding("It earned \u00bd of the spread.", OVERVIEW)

    assert not spelled.grounded and not odd.grounded
    assert "hundred" in spelled.unsupported[0].text.lower()


def test_a_half_may_be_rounded_either_way() -> None:
    payload = {"tab": "trading", "kpis": {"pnl_eur": 401.25}}

    # Python rounds 401.25 to 401.2; a person may write either.
    for prose in ("It earned 401.2 EUR.", "It earned 401.3 EUR."):
        assert grounding.check_grounding(prose, payload).grounded, prose


def test_the_check_cannot_catch_a_real_number_used_in_the_wrong_place() -> None:
    # The battery runs 2 hours and cycles 1.71 times a day. A briefing saying it
    # cycled 2 times a day is wrong, and this check passes it: the figure is in
    # the payload, and nothing here knows which key a sentence is talking about.
    # It catches figures that exist nowhere in the data, which is its purpose.
    assert grounding.check_grounding("It cycled 2 times a day.", TRADING).grounded


def test_a_question_loses_its_figures_whether_written_in_digits_or_words() -> None:
    digits = grounding.strip_numerals("Why did it only earn 4242 EUR on 31 August?")
    words = grounding.strip_numerals("Why did it earn four thousand euros?")

    assert not any(character.isdigit() for character in digits)
    assert "August" in digits and "EUR" in digits
    assert "thousand" not in words and "four" not in words


@pytest.mark.parametrize("tab", list(PAYLOADS))
def test_every_template_briefing_is_grounded_by_construction(tab: str) -> None:
    briefing = provider_mod.TemplateProvider().write(PAYLOADS[tab])

    report = grounding.check_grounding(briefing.text, PAYLOADS[tab])

    assert briefing.provider == "template" and briefing.model is None
    assert report.grounded, [claim.text for claim in report.unsupported]
    # Sentences, not decimal points: 8,755.24 must not count as two stops.
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", briefing.text) if part]
    assert 1 <= len(sentences) <= 5


def test_an_unsettled_day_says_so_instead_of_guessing() -> None:
    unsettled = OVERVIEW | {
        "forecast": {"periods": 96, "prices_published": False},
        "dispatch": dict(OVERVIEW["dispatch"]),
    }

    text = provider_mod.TemplateProvider().write(unsettled).text

    assert "not published" in text and "not settled" in text
    assert "earned" not in text


def test_a_follow_up_question_is_answered_from_the_same_payload() -> None:
    briefing = provider_mod.TemplateProvider().write(
        OVERVIEW, provider_mod.FOLLOW_UPS["why_this_dispatch"]
    )

    assert "why this dispatch" in briefing.text.lower()
    assert grounding.check_grounding(briefing.text, OVERVIEW).grounded


def test_what_changed_is_answered_from_yesterdays_own_figures() -> None:
    payload = OVERVIEW | {
        "previous_day": {
            "day": "2026-09-13",
            "pnl_eur": 1032.11,
            "perfect_foresight_pnl_eur": 1100.4,
            "pnl_change_eur": -67.65,
            "pnl_change_size_eur": 67.65,
        }
    }

    briefing = provider_mod.TemplateProvider().write(
        payload, provider_mod.FOLLOW_UPS["what_changed"]
    )

    assert "1,032.11 EUR, so this day is 67.65 EUR less" in briefing.text
    assert grounding.check_grounding(briefing.text, payload).grounded


def test_what_changed_declines_when_there_was_no_day_before() -> None:
    briefing = provider_mod.TemplateProvider().write(
        OVERVIEW | {"previous_day": None}, provider_mod.FOLLOW_UPS["what_changed"]
    )

    assert "the payload holds only the figures above" in briefing.text
    assert grounding.check_grounding(briefing.text, OVERVIEW).grounded


def test_an_unknown_tab_has_no_briefing() -> None:
    with pytest.raises(provider_mod.NarrationError, match="no briefing"):
        provider_mod.TemplateProvider().write({"tab": "pnl"})


def test_the_key_is_read_from_the_environment_then_the_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('OTHER=1\nOPENAI_API_KEY="from-file"\n', encoding="utf-8")
    monkeypatch.delenv(provider_mod.API_KEY_ENV, raising=False)

    assert provider_mod.load_api_key(env_file) == "from-file"

    monkeypatch.setenv(provider_mod.API_KEY_ENV, "from-environment")
    assert provider_mod.load_api_key(env_file) == "from-environment"
    assert provider_mod.load_api_key(tmp_path / "absent.env") == "from-environment"


def test_without_a_key_the_template_writes_and_with_one_the_model_would(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(provider_mod.API_KEY_ENV, raising=False)
    empty = tmp_path / "none.env"

    chosen = provider_mod.build_provider(empty)
    assert isinstance(chosen, provider_mod.TemplateProvider)

    monkeypatch.setenv(provider_mod.API_KEY_ENV, "test-key")
    with_key = provider_mod.build_provider(empty)
    # The model is selected, but nothing calls it here.
    assert isinstance(with_key, provider_mod.OpenAIProvider)
    assert with_key.name == "openai" and with_key.model == provider_mod.DEFAULT_MODEL
