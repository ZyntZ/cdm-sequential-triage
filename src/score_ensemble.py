"""Fixed score combinations for development-only ensemble diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd

SCORE_METHODS = ("arithmetic_mean", "geometric_mean", "maximum", "minimum")


def combine_scores(
    frame: pd.DataFrame,
    left_col: str = "catboost_snapshot",
    right_col: str = "catboost_tail_aligned",
) -> pd.DataFrame:
    """Add pre-specified combinations while preserving lower-is-safer semantics.

    ``maximum`` is pointwise pessimistic and ``minimum`` is pointwise
    optimistic. Once each score is independently recalibrated, however, their
    resulting decision sets need not remain subsets or supersets.
    """
    missing = {left_col, right_col}.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing score columns: {sorted(missing)}")
    left = pd.to_numeric(frame[left_col], errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(frame[right_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Scores must be finite")
    if ((left < 0) | (left > 1) | (right < 0) | (right > 1)).any():
        raise ValueError("Scores must lie in [0, 1]")
    result = frame.copy()
    result["arithmetic_mean"] = (left + right) / 2.0
    result["geometric_mean"] = np.sqrt(left * right)
    result["maximum"] = np.maximum(left, right)
    result["minimum"] = np.minimum(left, right)
    return result
