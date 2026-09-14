"""Formatting helpers of the Phase 4 results report."""

from __future__ import annotations

import pandas as pd

from src.trading.phase4_report import markdown_table


def test_markdown_table_formats_each_column() -> None:
    frame = pd.DataFrame({"name": ["median", "q25"], "capture": [0.901, 0.853]})
    lines = markdown_table(
        frame, [("name", "strategy", "{}"), ("capture", "capture", "{:.1%}")]
    )
    assert lines == [
        "| strategy | capture |",
        "|---|---|",
        "| median | 90.1% |",
        "| q25 | 85.3% |",
    ]
