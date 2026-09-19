import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_schema import outlier_bounds  # noqa: E402


def _sessions(session_code: str, null_pcts: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "file": [Path(f"r{i:02d}.parquet") for i in range(len(null_pcts))],
        "session_code": session_code,
        "missing": [[] for _ in null_pcts],
        "null_pct": null_pcts,
    })


def test_tukey_bound_does_not_flag_the_normal_q_cluster():
    # Shape mirrors what came back from real data: Q sessions tightly
    # clustered 35-43%, one genuine outlier at 60.9% (Baku 2025).
    normal_q = [0.382, 0.407, 0.392, 0.391, 0.422, 0.367, 0.405, 0.365, 0.364,
                0.377, 0.366, 0.353, 0.362, 0.435, 0.352, 0.391, 0.376, 0.368,
                0.366, 0.408, 0.360, 0.357, 0.413, 0.381, 0.365, 0.381, 0.397,
                0.362, 0.366, 0.400, 0.373, 0.408, 0.384, 0.375, 0.390, 0.354]
    sessions = _sessions("Q", normal_q + [0.609])
    bounds = outlier_bounds(sessions)

    # The bound must sit strictly between the normal cluster's max and the
    # real outlier, so the cluster passes and the outlier gets flagged.
    assert max(normal_q) < bounds["Q"] < 0.609


def test_tukey_bound_flags_only_the_true_outlier():
    normal_q = [0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.36]
    sessions = _sessions("Q", normal_q + [0.609])
    bounds = outlier_bounds(sessions)

    flagged = sessions[sessions["null_pct"] > bounds["Q"]]
    assert len(flagged) == 1
    assert flagged.iloc[0]["null_pct"] == 0.609


def test_min_flag_threshold_floors_a_too_tight_distribution():
    # If every sample in a type is nearly identical, the IQR collapses to
    # ~0 and a naive Tukey bound would flag tiny, meaningless noise. The
    # floor exists precisely to stop that.
    sessions = _sessions("SQ", [0.20, 0.201, 0.199, 0.20, 0.202])
    bounds = outlier_bounds(sessions)
    assert bounds["SQ"] >= 0.45
