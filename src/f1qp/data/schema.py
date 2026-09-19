"""Canonical lap-level schema + validation.

Phase 1 tasks: "Validate data schema" and "Document data quirks and
assumptions". This module produces a structured report instead of silently
dropping bad rows, so quirks get written down (docs/data_quirks.md, via
scripts/validate_schema.py) rather than discovered later during feature
engineering.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

REQUIRED_LAP_COLUMNS = [
    "Driver", "LapTime", "LapNumber", "Stint", "Compound", "TyreLife",
    "Sector1Time", "Sector2Time", "Sector3Time", "IsPersonalBest", "Deleted",
]

# A lap with no LapTime is normal, not a defect: out-laps, in-laps, and laps
# cut short by a red flag or safety car never complete a timed lap. Real
# data showed this is a BIGGER share than intuition suggests even for
# qualifying - every push lap has an out-lap and an in-lap bracketing it,
# so normal Q running lands around 35-43% null, not near zero. A first
# guess of "<35% for Q" flagged ~40 completely ordinary qualifying sessions
# across every season - a fixed number picked without looking at the data
# was wrong twice in a row, which is exactly why this is only a coarse net.
#
# These bounds exist ONLY for the one-session-at-a-time check at download
# time, when there's no dataset yet to compare against - they're set loose
# enough to avoid false alarms during a long backfill, catching only
# something obviously dead (a session that's almost nothing but red flag).
# The real anomaly detection - Tukey's rule against the actual distribution
# of everything downloaded - lives in scripts/validate_schema.py, which
# runs after a batch and has real data to compare each session against.
EXPECTED_MAX_NULL_LAPTIME_PCT = {
    "FP1": 0.75, "FP2": 0.75, "FP3": 0.75, "SQ": 0.75,
    "Q": 0.55,
}
DEFAULT_MAX_NULL_LAPTIME_PCT = 0.75


@dataclass
class SchemaReport:
    year: int
    round_number: int
    session_code: str
    missing_columns: list[str] = field(default_factory=list)
    null_lap_time_pct: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        if self.missing_columns:
            return False
        limit = EXPECTED_MAX_NULL_LAPTIME_PCT.get(self.session_code, DEFAULT_MAX_NULL_LAPTIME_PCT)
        return self.null_lap_time_pct < limit


def validate_laps(laps: pd.DataFrame, *, year: int, round_number: int, session_code: str) -> SchemaReport:
    report = SchemaReport(year=year, round_number=round_number, session_code=session_code)
    report.missing_columns = [c for c in REQUIRED_LAP_COLUMNS if c not in laps.columns]

    if "LapTime" in laps.columns and len(laps):
        report.null_lap_time_pct = float(laps["LapTime"].isna().mean())

    if session_code in ("Q", "SQ") and "Driver" in laps.columns:
        # A full field is 20 drivers; well below that usually means a red
        # flag, a weather-shortened session, or a DNS/DNQ worth a note in
        # docs/data_quirks.md rather than a silent NaN downstream.
        n_drivers = laps["Driver"].nunique()
        if n_drivers < 15:
            report.notes.append(f"Only {n_drivers} drivers have laps - check for a red flag or DNS/DNQ.")

    return report
