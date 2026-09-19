"""Tests for extract_qualifying_targets.py's `_needs_fetch` - the Sep 13
2026 fix for the "check for official result" button's Space performance
problem: this script re-fetches EVERY event in `_events()` from FastF1
over the network every run, and on a Hugging Face Space (cold FastF1
cache - the big cache is training-only, never baked into the image) that
used to mean re-downloading years of historical qualifying results just
to check one new round's result, easily overrunning data_fetch.py's own
900s per-step timeout. See extract_qualifying_targets.py's own module
docstring ("Incremental when --round is passed") for the full story.

Imports `extract_qualifying_targets` directly (not
`scripts.extract_qualifying_targets`) - same `scripts/`-on-pythonpath
convention as tests/test_build_features_merge.py.
"""

from __future__ import annotations

from extract_qualifying_targets import _needs_fetch


def test_bare_invocation_always_fetches_everything():
    """extra_rounds=[] (the bare `python scripts/extract_qualifying_targets.py`
    invocation) must behave EXACTLY as before this fix: main() never
    populates existing_keys in that case, so this always returns True
    regardless of what's passed as existing_keys here."""
    assert _needs_fetch(2023, 5, extra_rounds=[], existing_keys=set()) is True
    assert _needs_fetch(2023, 5, extra_rounds=[], existing_keys={(2023, 5)}) is True


def test_incremental_skips_an_already_cached_event_not_explicitly_requested():
    assert _needs_fetch(2023, 5, extra_rounds=[13], existing_keys={(2023, 5), (2026, 12)}) is False


def test_incremental_always_refetches_an_explicitly_requested_round():
    """Damiano clicking "check for official result" on Round 13 must
    always actually contact FastF1 for Round 13 - even if a (possibly
    stale/pre-quali) Round 13 row already happens to exist on disk."""
    assert _needs_fetch(2026, 13, extra_rounds=[13], existing_keys={(2026, 13)}) is True


def test_incremental_fetches_an_uncached_event_that_was_not_requested():
    """A round in `_events()` that ISN'T cached yet (e.g. this is the
    very first incremental run ever) still gets fetched, not silently
    skipped - `existing_keys` only ever suppresses a refetch for
    something that's genuinely already there."""
    assert _needs_fetch(2026, 12, extra_rounds=[13], existing_keys=set()) is True
