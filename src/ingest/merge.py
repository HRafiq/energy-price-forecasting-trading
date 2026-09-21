"""Laying a freshly downloaded window over the file that is already there.

A windowed download fetches only the days a run needs. The file it writes to may
hold years, because the backtest reads it, so the window is merged in rather than
written over. Rows the window has win, so a value revised at source replaces the
one it revises, and rows outside the window are kept as they were.

The columns must match. A window built after a column was added or removed would
otherwise concatenate into a frame with a cliff of NaN at the boundary, which
every later step reads as real missing data instead of a mistake.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["merge_into"]


def merge_into(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """``frame`` laid over the table at ``path``, if there is one."""
    if not path.exists():
        return frame
    existing = pd.read_parquet(path)
    if list(existing.columns) != list(frame.columns):
        added = sorted(set(frame.columns) - set(existing.columns))
        dropped = sorted(set(existing.columns) - set(frame.columns))
        raise ValueError(
            f"{path} does not have the columns this download built"
            + (f"; added {added}" if added else "")
            + (f"; missing {dropped}" if dropped else "")
            + ". Rebuild it in full rather than merging a window into it"
        )
    kept = existing.loc[~existing.index.isin(frame.index)]
    if kept.empty:
        return frame
    return pd.concat([kept, frame]).sort_index()
