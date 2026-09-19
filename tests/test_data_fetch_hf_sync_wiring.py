"""Tests that f1qp.serving.data_fetch's start_fetch_job / start_results_job
push the file they just rebuilt out to f1qp.serving.hf_sync once their
steps succeed (Sep 13 2026 addition - see hf_sync's own module docstring
and data_fetch.py's updated docstrings on start_fetch_job/
start_results_job).

Never runs the real scripts/download_2026.py etc. subprocesses - those
need real FastF1 network access. Instead monkeypatches
f1qp.serving.data_fetch.subprocess.run itself to fake a successful
(returncode 0) run for every step, so _run_step/_worker's own control
flow (including the on_success callback this file is actually testing)
runs for real.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import f1qp.serving.data_fetch as data_fetch_mod
from f1qp.serving.data_fetch import job_status, start_fetch_job, start_results_job


def _fake_subprocess_run(cmd, **kwargs):
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def _wait_for_job(job_id, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = job_status(job_id)
        if job["status"] != "running":
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never finished within {timeout}s")


def test_start_fetch_job_pushes_features_parquet_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(data_fetch_mod.subprocess, "run", _fake_subprocess_run)
    features_path = tmp_path / "features.parquet"
    features_path.write_bytes(b"fake-parquet-bytes")
    monkeypatch.setenv("F1QP_FEATURES_PATH", str(features_path))

    calls = []
    monkeypatch.setattr(
        data_fetch_mod.hf_sync, "push_file", lambda path, path_in_repo: calls.append((path, path_in_repo))
    )

    job_id = start_fetch_job(2026, 13)
    job = _wait_for_job(job_id)

    assert job["status"] == "done"
    assert calls == [(features_path, "processed/features.parquet")]


def test_start_results_job_pushes_qualifying_targets_parquet_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(data_fetch_mod.subprocess, "run", _fake_subprocess_run)
    targets_path = tmp_path / "qualifying_targets.parquet"
    targets_path.write_bytes(b"fake-parquet-bytes")
    monkeypatch.setenv("F1QP_TARGETS_PATH", str(targets_path))

    calls = []
    monkeypatch.setattr(
        data_fetch_mod.hf_sync, "push_file", lambda path, path_in_repo: calls.append((path, path_in_repo))
    )

    job_id = start_results_job(2026, 13)
    job = _wait_for_job(job_id)

    assert job["status"] == "done"
    assert calls == [(targets_path, "processed/qualifying_targets.parquet")]


def test_on_success_is_never_called_when_a_step_fails(monkeypatch):
    def _failing_run(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(data_fetch_mod.subprocess, "run", _failing_run)
    calls = []
    monkeypatch.setattr(
        data_fetch_mod.hf_sync, "push_file", lambda path, path_in_repo: calls.append((path, path_in_repo))
    )

    job_id = start_fetch_job(2026, 13)
    job = _wait_for_job(job_id)

    assert job["status"] == "error"
    assert calls == []
