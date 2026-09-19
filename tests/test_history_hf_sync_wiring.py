"""Tests that f1qp.serving.history.record_launch actually wires up
f1qp.serving.hf_sync (Sep 13 2026 addition - see hf_sync's own module
docstring). Kept as a separate file from tests/test_history.py on purpose
- it only adds coverage for the new wiring, without touching that file's
existing tests at all.

record_launch fires the push in a background daemon thread (a Launch
should feel instant - see its own comment), so these tests monkeypatch
f1qp.serving.history.hf_sync.push_file with a stub that sets a
threading.Event, then wait on that event with a short timeout instead of
sleeping a fixed amount or asserting immediately.
"""

from __future__ import annotations

import threading

import f1qp.serving.history as history_mod
from f1qp.serving.history import record_launch
from f1qp.serving.predict import DriverPrediction


def _pred(driver="VER") -> DriverPrediction:
    return DriverPrediction(
        driver=driver,
        era=1,
        is_sprint=False,
        n_practice_sessions=3,
        predicted_quali_time_seconds=90.0,
        interval_low_seconds=89.0,
        interval_high_seconds=91.0,
        interval_width_seconds=2.0,
        interval_level_pct=50.0,
        interval_exact=True,
    )


def test_record_launch_pushes_the_written_file_to_hf_sync_in_the_background(tmp_path, monkeypatch):
    launches_dir = tmp_path / "launches"
    calls = []
    done = threading.Event()

    def _fake_push_file(path, path_in_repo):
        calls.append((path, path_in_repo))
        done.set()

    monkeypatch.setattr(history_mod.hf_sync, "push_file", _fake_push_file)

    record_launch(
        year=2026,
        round_number=13,
        predictions=[_pred()],
        excluded_test_drivers=[],
        model_trained_at_utc="2026-08-24T21:23:34+00:00",
        launched_at_utc="2026-09-13T12:00:00+00:00",
        launches_dir=launches_dir,
    )

    assert done.wait(timeout=2.0), "hf_sync.push_file was never called"
    assert len(calls) == 1
    path, path_in_repo = calls[0]
    assert path == launches_dir / "2026_13.json"
    assert path_in_repo == "predictions/launches/2026_13.json"
    assert path.exists()  # the local write itself must never depend on the push succeeding


def test_record_launch_still_writes_locally_even_if_hf_sync_push_raises(tmp_path, monkeypatch):
    """push_file itself is supposed to swallow its own errors (see
    hf_sync's tests), but this asserts the local write - the source of
    truth for the HTTP response - can never be affected by it either way,
    even in the worst case where the stub itself misbehaves."""
    launches_dir = tmp_path / "launches"
    done = threading.Event()

    def _raising_push_file(path, path_in_repo):
        done.set()
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(history_mod.hf_sync, "push_file", _raising_push_file)

    record_launch(
        year=2026,
        round_number=14,
        predictions=[_pred()],
        excluded_test_drivers=[],
        model_trained_at_utc="2026-08-24T21:23:34+00:00",
        launched_at_utc="2026-09-13T12:05:00+00:00",
        launches_dir=launches_dir,
    )

    assert done.wait(timeout=2.0)
    assert (launches_dir / "2026_14.json").exists()
