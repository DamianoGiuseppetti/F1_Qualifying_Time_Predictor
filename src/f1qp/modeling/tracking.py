"""Phase 5: lightweight MLflow tracking for scripts/retrain_pipeline.py.

The project brief lists "Monitoring: MLflow lite (experiment tracking,
model versioning)" and the deployment checklist has its own separate line,
"Model versioning in MLflow (v1.0)" - distinct from the models/lstm/
history/<timestamp>/ archiving retrain_pipeline.py already does (Phase 3/4).
That archive stays exactly as-is; this module adds MLflow ON TOP of it,
so the checklist item has a real, working home instead of only file-based
versioning standing in for "MLflow lite" in spirit.

**Design, deliberately small**:
- Local SQLite-backed tracking (`sqlite:///<repo_root>/mlruns/mlflow.db`),
  with run artifacts (the logged model, metadata/conformal json files)
  written under `<repo_root>/mlruns/artifacts/` - no MLflow server to run
  or configure, still fully local, consistent with every other artifact in
  this project living on disk under `repo/`. **Correction (Aug 25 2026,
  first real retrain_pipeline.py run)**: this was originally the raw
  filesystem store (`file:<repo_root>/mlruns`), which failed on Damiano's
  machine with `MlflowException("The filesystem tracking backend ... is
  in maintenance mode and will not receive further updates ... migrate to
  a database backend")` - the installed mlflow version (requirements.txt
  pins only a floor, `mlflow>=2.14`, so pip resolved whatever is newest)
  has deprecated FileStore for new tracking stores. Fixed by switching to
  SQLAlchemyStore over a local SQLite file instead - MLflow's own
  recommended migration path, and (as a second, independent reason) the
  Model Registry `mlflow.pytorch.log_model(..., registered_model_name=...)`
  call below needs a database-backed store to fully work, which FileStore
  was never guaranteed to support in every mlflow version either. Both the
  sqlite file and the artifacts directory live under `mlruns/`, which
  `.gitignore` already ignored (added ahead of this, Phase 3) - both are
  regenerable, never meant to be committed. If Damiano has a stale
  `mlruns/` directory from the failed first attempt, it's safe to delete
  before the next retrain - this module recreates everything it needs.
- One MLflow run per `retrain_pipeline.py` execution, under a single
  experiment (`EXPERIMENT_NAME`).
- Registers the retrained LSTM under one registered model name
  (`REGISTERED_MODEL_NAME`) every run, via `mlflow.pytorch.log_model(...,
  registered_model_name=...)` - MLflow bumps the version automatically
  (v1, v2, v3, ...) each time this succeeds, which is literally what
  "Model versioning in MLflow (v1.0)" asks for.
- **Best-effort, never blocks a retrain**: if `mlflow` isn't installed, or
  anything about logging fails (a locked file, a corrupt tracking dir,
  whatever), `log_retrain_run` prints a warning and returns `None` rather
  than raising - retraining the production model is the important side
  effect of running that pipeline; tracking it is secondary and must never
  be the reason a retrain fails.

**Correction #2 (Aug 25 2026, first successful post-SQLite-fix retrain_pipeline.py
run)**: params/metrics logged fine this time, but model registration itself
still failed - `mlflow.pytorch.log_model`'s `serialization_format` defaults
to `"pt2"` as of this installed mlflow version (3.15.1), which saves the
model via `torch.export.save` (a traced graph) and REQUIRES a concrete
`input_example` (a bare numpy array/tensor, or a tuple/list of them) to
trace `model.forward` with. This project's LSTM forward signature takes a
padded/packed variable-length sequence plus static context, not a single
plain tensor - not something worth constructing a synthetic trace input for
just to satisfy a serialization format we don't need. Fixed by pinning
`serialization_format="pickle"` explicitly (mlflow's other supported
option - the pre-3.x default, cloudpickle-based, no `input_example`
required, and works with an arbitrary forward signature). Pickle's usual
caveat (arbitrary code execution on load) is a non-issue here: this is a
single local user's own model on their own machine, never loading a model
file from anyone else.

Read back via `f1qp.modeling.tracking.search_runs()` (used by
scripts/retrain_pipeline.py's own before/after summary if needed) or the
Streamlit dashboard's "Model performance history" tab (dashboard/app.py),
which reimplements the same sqlite lookup inline rather than importing
this module - see that file's own docstring for why - and must be kept in
sync with the tracking-store layout here if it ever changes again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

REPO_ROOT = Path(__file__).resolve().parents[3]
MLRUNS_DIR = REPO_ROOT / "mlruns"
EXPERIMENT_NAME = "f1_qualifying_predictor"
REGISTERED_MODEL_NAME = "f1_qualifying_lstm"


def _sqlite_tracking_uri(tracking_dir: Path) -> str:
    return f"sqlite:///{(tracking_dir / 'mlflow.db').resolve()}"


def _configure_mlflow(tracking_dir: Path) -> None:
    """Point mlflow at `tracking_dir`'s SQLite store + artifacts folder,
    creating both (and the experiment, if this is the very first run)."""
    import mlflow

    tracking_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = tracking_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(_sqlite_tracking_uri(tracking_dir))
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=artifacts_dir.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_retrain_run(
    *,
    params: Dict,
    metrics: Dict,
    artifact_paths: Optional[Sequence[Union[str, Path]]] = None,
    model=None,
    mlruns_dir: Optional[Path] = None,
) -> Optional[str]:
    """Log one retrain_pipeline.py run to the local MLflow tracking store.

    `params`: run-level params (n_train, epoch_count, ...) - values that
    describe the run's configuration, not a measured outcome.
    `metrics`: measured outcomes (pooled_mape, pooled_r2, interval widths,
    ...). `None` values are dropped before logging (MLflow's log_metrics
    doesn't accept them).
    `artifact_paths`: existing files to attach to the run for lineage
    (e.g. final_model_metadata.json, conformal_intervals.json) - paths
    that don't exist are silently skipped rather than raising.
    `model`: the trained torch model to register, or `None` to log
    params/metrics only without registering a model version.

    Returns the MLflow run_id on success, or `None` if mlflow isn't
    installed or logging failed for any other reason - see module
    docstring for why this never raises.
    """
    try:
        import mlflow
    except ImportError:
        print(
            "mlflow is not installed - skipping experiment tracking / model "
            "versioning for this run (pip install mlflow, or `pip install -r "
            "requirements.txt`, to enable it). The models/lstm/history/ "
            "archive above is unaffected.",
            flush=True,
        )
        return None
    except Exception as exc:
        print(f"Could not import mlflow ({exc!r}) - continuing without tracking.", flush=True)
        return None

    tracking_dir = Path(mlruns_dir) if mlruns_dir is not None else MLRUNS_DIR

    try:
        _configure_mlflow(tracking_dir)

        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
            for path in artifact_paths or []:
                if Path(path).exists():
                    mlflow.log_artifact(str(path))
            if model is not None:
                # `mlflow.pytorch` imports torch - deliberately imported
                # HERE, not at module top, and in its own try/except: a
                # broken torch install (observed for real while verifying
                # this module - an OSError loading a CUDA shared library)
                # should only cost the model-registration step, not the
                # params/metrics/artifact logging above, which has nothing
                # to do with torch and already succeeded by this point.
                try:
                    import mlflow.pytorch

                    mlflow.pytorch.log_model(
                        model,
                        "model",
                        registered_model_name=REGISTERED_MODEL_NAME,
                        # Pin explicitly - this mlflow version's default
                        # ("pt2") requires a concrete input_example to trace
                        # model.forward, which this LSTM's variable-length/
                        # packed-sequence signature isn't a good fit for.
                        # See module docstring's "Correction #2".
                        serialization_format="pickle",
                    )
                except Exception as exc:
                    print(
                        f"Could not register model version ({exc!r}) - this run's "
                        f"params/metrics were still logged.",
                        flush=True,
                    )
            return run.info.run_id
    except Exception as exc:  # tracking is best-effort, never blocks a retrain
        print(f"MLflow logging failed ({exc!r}) - continuing without it.", flush=True)
        return None


def search_runs(mlruns_dir: Optional[Path] = None):
    """Read back every logged run for the dashboard's "Model performance
    history" tab. Returns an empty-shaped `pandas.DataFrame` (not `None`
    or a raised error) if mlflow isn't installed, no run has ever been
    logged yet (no mlflow.db under `mlruns_dir`), or reading otherwise
    fails - callers can treat "no history yet" as one uniform, easy-to-
    render case.
    """
    import pandas as pd

    tracking_dir = Path(mlruns_dir) if mlruns_dir is not None else MLRUNS_DIR
    db_path = tracking_dir / "mlflow.db"
    if not db_path.exists():
        return pd.DataFrame()

    try:
        import mlflow
    except ImportError:
        return pd.DataFrame()

    try:
        mlflow.set_tracking_uri(_sqlite_tracking_uri(tracking_dir))
        runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
    except Exception:
        return pd.DataFrame()
    return runs if runs is not None else pd.DataFrame()
