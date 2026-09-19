"""Tests for build_features.py's `_carry_forward_missing` - the Sep 13
2026 fix for a real data-loss bug: this script used to rebuild
features.parquet from scratch every run using ONLY whatever raw laps
happen to be on disk, silently DROPPING every weekend it couldn't find
raw laps for. That's harmless on a full local checkout, but on a Hugging
Face Space (which never bakes in the huge, training-only raw-lap cache)
running this via the app's own on-demand fetch would wipe every other
weekend's rows on the very first click - see build_features.py's own
module docstring ("Carry-forward for missing raw laps") for the full
story.

Imports `build_features` directly (not `scripts.build_features`) because
`scripts/` has no `__init__.py` and is on pytest's pythonpath
(pyproject.toml, same convention tests/test_dashboard_formatting.py
already uses for `dashboard/` - see its own docstring).
"""

from __future__ import annotations

import pandas as pd

from build_features import _carry_forward_missing


def _rows(*triples):
    """triples: (year, round_number, val)."""
    return pd.DataFrame(
        [{"Year": y, "RoundNumber": r, "val": v} for y, r, v in triples]
    )


def test_no_existing_data_returns_rebuilt_unchanged():
    rebuilt = _rows((2026, 13, "new"))
    out = _carry_forward_missing(rebuilt, None)
    assert out is rebuilt


def test_empty_existing_data_returns_rebuilt_unchanged():
    rebuilt = _rows((2026, 13, "new"))
    out = _carry_forward_missing(rebuilt, pd.DataFrame())
    assert len(out) == 1
    assert out.iloc[0]["val"] == "new"


def test_missing_raw_laps_are_carried_forward_from_existing():
    """The core bug fix: rounds this run couldn't rebuild (no raw laps on
    disk - the Space's exact situation for every historical round) keep
    their previously-built rows instead of vanishing."""
    existing = _rows(
        (2023, 1, "old-2023-r1"),
        (2026, 12, "old-2026-r12"),
    )
    rebuilt = _rows((2026, 13, "new-2026-r13"))  # only R13's raw laps were on disk this run

    out = _carry_forward_missing(rebuilt, existing)

    assert len(out) == 3
    by_key = {(row["Year"], row["RoundNumber"]): row["val"] for _, row in out.iterrows()}
    assert by_key[(2023, 1)] == "old-2023-r1"
    assert by_key[(2026, 12)] == "old-2026-r12"
    assert by_key[(2026, 13)] == "new-2026-r13"


def test_a_freshly_rebuilt_round_wins_over_its_old_carried_forward_row():
    """A round that WAS just rebuilt (raw laps were found) always
    supersedes whatever it had before - a real refresh, not a gap to fill."""
    existing = _rows((2026, 12, "stale"))
    rebuilt = _rows((2026, 12, "fresh"))

    out = _carry_forward_missing(rebuilt, existing)

    assert len(out) == 1
    assert out.iloc[0]["val"] == "fresh"


def test_everything_missing_carries_forward_the_entire_existing_file():
    """The worst case a bare-metal Space run without this fix used to hit:
    zero raw laps found for anything requested. Every previously-built row
    now survives untouched instead of the output collapsing to nothing."""
    existing = _rows((2023, 1, "a"), (2023, 2, "b"), (2026, 12, "c"))
    rebuilt = pd.DataFrame(columns=["Year", "RoundNumber", "val"])

    out = _carry_forward_missing(rebuilt, existing)

    assert len(out) == 3
    assert set(zip(out["Year"], out["RoundNumber"])) == {(2023, 1), (2023, 2), (2026, 12)}
