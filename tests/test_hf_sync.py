"""Tests for f1qp.serving.hf_sync - the Hugging Face dataset-repo mirror
behind the Sep 13 2026 "storage should be into hugging face without be in
my local storage anymore" addition (see the module's own docstring).

`huggingface_hub` is imported lazily, INSIDE push_file/pull_all, so these
tests fake it out via sys.modules rather than requiring the real package
to be installed - the same reason hf_sync.py is written that way (a
deployment that never sets HF_DATASET_REPO/HF_TOKEN, i.e. every test in
this repo before this file, never needs the package at all). Every test
here explicitly sets or clears HF_DATASET_REPO/HF_TOKEN via monkeypatch
rather than relying on ambient environment state.
"""

from __future__ import annotations

import sys
import types

import pytest

import f1qp.serving.hf_sync as hf_sync


class _FakeHfApi:
    """Records every upload_file call it receives, so tests can assert on
    exactly what was pushed without touching the network."""

    calls: list[dict] = []

    def __init__(self, token=None):
        self.token = token

    def upload_file(self, **kwargs):
        _FakeHfApi.calls.append({"token": self.token, **kwargs})


def _install_fake_hub(monkeypatch, snapshot_download=None, raise_on_upload=None):
    _FakeHfApi.calls = []
    fake = types.ModuleType("huggingface_hub")

    if raise_on_upload is not None:
        class _RaisingHfApi(_FakeHfApi):
            def upload_file(self, **kwargs):
                raise raise_on_upload

        fake.HfApi = _RaisingHfApi
    else:
        fake.HfApi = _FakeHfApi

    fake.snapshot_download = snapshot_download or (lambda **kwargs: (_ for _ in ()).throw(RuntimeError("not used")))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    return fake


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("HF_DATASET_REPO", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)


def test_disabled_by_default():
    assert hf_sync.enabled() is False


def test_disabled_with_only_one_of_the_two_env_vars(monkeypatch):
    monkeypatch.setenv("HF_DATASET_REPO", "someone/some-dataset")
    assert hf_sync.enabled() is False
    monkeypatch.delenv("HF_DATASET_REPO")
    monkeypatch.setenv("HF_TOKEN", "tok")
    assert hf_sync.enabled() is False


def test_enabled_once_both_env_vars_set(monkeypatch):
    monkeypatch.setenv("HF_DATASET_REPO", "someone/some-dataset")
    monkeypatch.setenv("HF_TOKEN", "tok")
    assert hf_sync.enabled() is True


def test_push_file_noop_when_disabled(tmp_path, monkeypatch):
    _install_fake_hub(monkeypatch)
    local = tmp_path / "features.parquet"
    local.write_text("data")
    hf_sync.push_file(local, "processed/features.parquet")
    assert _FakeHfApi.calls == []


def test_push_file_noop_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASET_REPO", "someone/some-dataset")
    monkeypatch.setenv("HF_TOKEN", "tok")
    _install_fake_hub(monkeypatch)
    missing = tmp_path / "does_not_exist.json"
    hf_sync.push_file(missing, "predictions/launches/2026_13.json")
    assert _FakeHfApi.calls == []


def test_push_file_uploads_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASET_REPO", "someone/some-dataset")
    monkeypatch.setenv("HF_TOKEN", "s3cr3t")
    _install_fake_hub(monkeypatch)
    local = tmp_path / "2026_13.json"
    local.write_text('{"year": 2026}')

    hf_sync.push_file(local, "predictions/launches/2026_13.json")

    assert len(_FakeHfApi.calls) == 1
    call = _FakeHfApi.calls[0]
    assert call["token"] == "s3cr3t"
    assert call["path_or_fileobj"] == str(local)
    assert call["path_in_repo"] == "predictions/launches/2026_13.json"
    assert call["repo_id"] == "someone/some-dataset"
    assert call["repo_type"] == "dataset"


def test_push_file_swallows_upload_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASET_REPO", "someone/some-dataset")
    monkeypatch.setenv("HF_TOKEN", "tok")
    _install_fake_hub(monkeypatch, raise_on_upload=RuntimeError("network is down"))
    local = tmp_path / "features.parquet"
    local.write_text("data")

    # Must not raise - a durability sync failing can never break the
    # request (a Launch, a data-fetch job) that triggered it.
    hf_sync.push_file(local, "processed/features.parquet")


def test_pull_all_noop_when_disabled(tmp_path, monkeypatch):
    called = {"snapshot_download": False}

    def _snapshot_download(**kwargs):
        called["snapshot_download"] = True
        return str(tmp_path)

    _install_fake_hub(monkeypatch, snapshot_download=_snapshot_download)
    hf_sync.pull_all(
        predictions_dir=tmp_path / "launches",
        features_path=tmp_path / "features.parquet",
        targets_path=tmp_path / "targets.parquet",
    )
    assert called["snapshot_download"] is False


def test_pull_all_copies_files_from_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASET_REPO", "someone/some-dataset")
    monkeypatch.setenv("HF_TOKEN", "tok")

    snapshot_root = tmp_path / "snapshot"
    (snapshot_root / "predictions" / "launches").mkdir(parents=True)
    (snapshot_root / "predictions" / "launches" / "2026_13.json").write_text('{"year": 2026}')
    (snapshot_root / "processed").mkdir(parents=True)
    (snapshot_root / "processed" / "features.parquet").write_bytes(b"features-bytes")
    (snapshot_root / "processed" / "qualifying_targets.parquet").write_bytes(b"targets-bytes")

    captured_kwargs = {}

    def _snapshot_download(**kwargs):
        captured_kwargs.update(kwargs)
        return str(snapshot_root)

    _install_fake_hub(monkeypatch, snapshot_download=_snapshot_download)

    predictions_dir = tmp_path / "dest" / "launches"
    features_path = tmp_path / "dest" / "processed" / "features.parquet"
    targets_path = tmp_path / "dest" / "processed" / "qualifying_targets.parquet"

    hf_sync.pull_all(predictions_dir=predictions_dir, features_path=features_path, targets_path=targets_path)

    assert captured_kwargs["repo_id"] == "someone/some-dataset"
    assert captured_kwargs["repo_type"] == "dataset"
    assert captured_kwargs["token"] == "tok"

    assert (predictions_dir / "2026_13.json").read_text() == '{"year": 2026}'
    assert features_path.read_bytes() == b"features-bytes"
    assert targets_path.read_bytes() == b"targets-bytes"


def test_pull_all_swallows_download_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASET_REPO", "someone/some-dataset")
    monkeypatch.setenv("HF_TOKEN", "tok")

    def _snapshot_download(**kwargs):
        raise RuntimeError("repo does not exist yet")

    _install_fake_hub(monkeypatch, snapshot_download=_snapshot_download)

    # Must not raise - a fresh/never-pushed-to dataset repo is a normal
    # state (the very first deploy), not an error.
    hf_sync.pull_all(
        predictions_dir=tmp_path / "launches",
        features_path=tmp_path / "features.parquet",
        targets_path=tmp_path / "targets.parquet",
    )
    assert not (tmp_path / "launches").exists()
