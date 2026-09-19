"""Phase 4/5 groundwork: reusable "retrain after a completed round" logic.

Damiano's Aug 24 2026 direction, ahead of Phase 4: the deployed
application needs to check each GP's real results after the fact and
retrain the model with the new data - both because more data helps in
general, and specifically because 2026 is a new regulation era, so real
2026 data is the thing this model most needs more of. Decided (via
AskUserQuestion): build the retrain+recalibrate LOGIC now as a reusable,
self-contained script (scripts/retrain_pipeline.py); wiring it to an
actual automatic trigger (a Phase 4 API endpoint, a scheduled job, or
staying a manual step) is an open decision deferred until Phase 4 exists.

This module holds only the two pieces of PURE, reusable, fully-testable
logic that retrain_pipeline.py needs and that are worth keeping separate
from that script's I/O-heavy orchestration:

1. **`select_final_epoch_count`**: the first production model's fixed
   epoch count (8) was chosen by hand, one time, by eyeballing where the
   standalone 80/20 run's own best_epoch and the original LORO check's
   11 per-fold best_epoch values agreed (see scripts/train_final_lstm.py's
   original docstring). That reasoning needs to become a REPEATABLE RULE
   now that this happens every time a new round's data lands, not a
   one-off judgment call: take the MAXIMUM best_epoch observed across the
   (now larger, as rounds accumulate) set of LORO folds - the upper end
   actually observed to still help (never hurt) in every real fold, not
   the most common single value (which would ignore folds that
   benefited from more epochs).

2. **`format_retrain_comparison`**: a plain-text before/after summary
   (n_train, LORO pooled MAPE/R2, chosen epoch count, 50% conformal
   interval width) comparing the previous production artifacts against
   this run's fresh ones. This is the actual evidence for whether
   retraining is helping as the 2026 season provides more data - the
   whole point of running this pipeline repeatedly rather than once.

Era policy (Aug 24 2026 decision, revisit later): retrain_pipeline.py
always blends era 0 (2023-2025) and era 1 (2026) data - it does not
attempt era-1-only training even as more 2026 rounds accumulate, since
era 0 still provides far more volume (1,364 rows) than era 1 does today
(238 rows across 11 rounds). `era` remains a plain input feature the
model itself learns to weight; nothing here changes that policy - it's
just documented here since this module is where that kind of pipeline-
level decision lives.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def select_final_epoch_count(per_round_best_epochs: List[int]) -> int:
    """Derive `train_final_model`'s fixed epoch count from a fresh
    `leave_one_round_out_cv_lstm` run's per-fold `best_epoch` values.

    Rule: the maximum best_epoch observed across all folds. See module
    docstring for why this generalizes the original one-off manual choice
    (FINAL_TRAIN_EPOCHS=8) into something this pipeline can re-derive
    every time it runs, as the number and identity of folds changes with
    each newly completed round.

    Raises if given an empty list - can't derive a rule from zero folds.
    """
    if not per_round_best_epochs:
        raise ValueError("Need at least 1 fold's best_epoch to derive a final epoch count")
    return int(max(per_round_best_epochs))


_COMPARISON_FIELDS = [
    ("n_train", "{}"),
    ("pooled_mape", "{:.3f}%"),
    ("pooled_r2", "{:.3f}"),
    ("epoch_count", "{}"),
    ("interval_50pct", "{:.3f}s"),
]


def format_retrain_comparison(previous: Optional[Dict], current: Dict) -> str:
    """Plain-text before/after summary of one retrain pipeline run.

    `previous`/`current` are plain dicts with the same 5 keys: `n_train`,
    `pooled_mape`, `pooled_r2`, `epoch_count`, `interval_50pct` (the 50%
    split-conformal interval's era-1 deployment quantile, in seconds -
    the level actually shipped, see Task_List.txt). `previous=None` means
    this is the first-ever pipeline run - there's nothing to compare
    against yet, which is reported plainly rather than as an error (a
    missing prior run is an expected, normal state, not a bug).
    """
    if previous is None:
        return "No previous production run found - nothing to compare against yet."

    lines = ["Retrain comparison (previous -> current):"]
    for key, fmt in _COMPARISON_FIELDS:
        old_v, new_v = previous.get(key), current.get(key)
        old_str = fmt.format(old_v) if old_v is not None else "?"
        new_str = fmt.format(new_v) if new_v is not None else "?"
        lines.append(f"  {key}: {old_str} -> {new_str}")
    return "\n".join(lines)


def comparison_rows(previous: Optional[Dict], current: Dict) -> List[Dict[str, Any]]:
    """Structured (not preformatted-text) version of the same before/after
    comparison `format_retrain_comparison` prints - for a caller (the
    API's `GET /model/staged`, the Performance tab's staged-candidate
    card) that wants to render its own table rather than parse a string.
    Same 5 fields, same order; `previous` is None for a first-ever run
    (every row's `previous` comes back None, not an error)."""
    rows = []
    for key, fmt in _COMPARISON_FIELDS:
        rows.append({
            "key": key,
            "previous": previous.get(key) if previous else None,
            "current": current.get(key),
            "format": fmt,
        })
    return rows
