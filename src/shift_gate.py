"""Conformal applicability gate for fail-safe sequential triage.

The gate turns a multivariate feature vector into a scalar nonconformity score.
Its threshold is calibrated on an exchangeable event-level sample. Rows above
that threshold are outside the calibrated applicability region and must not
receive SAFE-EXCLUDE.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GateCalibration:
    threshold: float
    alpha: float
    rank: int
    n_calibration: int
    marginal_flag_bound: float


class ConformalShiftGate:
    """Robust max-deviation gate with split-conformal calibration.

    Location and scale are learned from a proper training sample. Calibration
    must use independent event-level rows. The score is the largest absolute
    robust z-score across configured numeric features. Missing and non-finite
    values receive an infinite score and are therefore blocked.
    """

    def __init__(self, feature_columns: list[str] | tuple[str, ...], min_scale: float = 1e-12):
        columns = tuple(feature_columns)
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("feature_columns must be non-empty and unique")
        if min_scale <= 0:
            raise ValueError("min_scale must be positive")
        self.feature_columns = columns
        self.min_scale = float(min_scale)
        self.location_: pd.Series | None = None
        self.scale_: pd.Series | None = None
        self.calibration_: GateCalibration | None = None

    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_columns).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing gate features: {sorted(missing)}")
        return frame.loc[:, self.feature_columns].apply(pd.to_numeric, errors="coerce")

    def fit(self, proper_training: pd.DataFrame) -> "ConformalShiftGate":
        x = self._matrix(proper_training)
        if x.empty:
            raise ValueError("proper_training must contain at least one row")
        if not np.isfinite(x.to_numpy(dtype=float)).all():
            raise ValueError("Proper-training gate features must be finite")
        location = x.median(axis=0)
        q75 = x.quantile(0.75)
        q25 = x.quantile(0.25)
        scale = (q75 - q25) / 1.349
        fallback = x.std(axis=0, ddof=0)
        scale = scale.where(scale > self.min_scale, fallback)
        scale = scale.where(scale > self.min_scale, 1.0)
        self.location_ = location.astype(float)
        self.scale_ = scale.astype(float)
        self.calibration_ = None
        return self

    def score(self, frame: pd.DataFrame) -> pd.Series:
        if self.location_ is None or self.scale_ is None:
            raise RuntimeError("Fit the gate before scoring")
        x = self._matrix(frame)
        values = ((x - self.location_).abs() / self.scale_).max(axis=1)
        finite_rows = np.isfinite(x.to_numpy(dtype=float)).all(axis=1)
        values = values.astype(float)
        values.loc[~finite_rows] = np.inf
        values.name = "shift_score"
        return values

    def calibrate(self, calibration_events: pd.DataFrame, alpha: float = 0.05) -> GateCalibration:
        """Calibrate a one-sided split-conformal upper threshold.

        Under event-level exchangeability, a new in-distribution event is
        flagged with marginal probability at most ``alpha``. If the requested
        level is unattainable for the calibration sample size, the threshold
        is infinite and the gate flags only non-finite inputs.
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        scores = self.score(calibration_events).to_numpy(dtype=float)
        if scores.size == 0:
            raise ValueError("calibration_events must contain at least one row")
        if not np.isfinite(scores).all():
            raise ValueError("Calibration gate features must be finite")
        n = scores.size
        rank = math.ceil((n + 1) * (1 - alpha))
        threshold = np.inf if rank > n else float(np.sort(scores)[rank - 1])
        calibration = GateCalibration(
            threshold=threshold,
            alpha=float(alpha),
            rank=int(rank),
            n_calibration=int(n),
            marginal_flag_bound=float((n + 1 - rank) / (n + 1)),
        )
        self.calibration_ = calibration
        return calibration

    def allows_safe_exclude(self, frame: pd.DataFrame) -> pd.Series:
        if self.calibration_ is None:
            raise RuntimeError("Calibrate the gate before applying it")
        scores = self.score(frame)
        allowed = np.isfinite(scores) & (scores <= self.calibration_.threshold)
        allowed.name = "shift_gate_allows_safe_exclude"
        return allowed
