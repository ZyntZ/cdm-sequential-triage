"""Runtime decision policy for sequential CDM triage.

The predictive model is intentionally kept outside this module.  It supplies a
score after each CDM; this module applies the frozen history, calibration and
applicability rules to that score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np
import pandas as pd

from shift_gate import ConformalShiftGate


class Decision(str, Enum):
    SAFE_EXCLUDE = "SAFE-EXCLUDE"
    MONITOR = "MONITOR"
    ESCALATE = "ESCALATE"


@dataclass(frozen=True)
class TriageDecision:
    event_id: Any
    sequence_number: int
    time_to_tca: float
    score: float
    decision: Decision
    reason: str
    shift_score: float | None
    shift_gate_allowed: bool
    decision_window_eligible: bool
    eligible_history_count: int

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["decision"] = self.decision.value
        return record


class SequentialTriagePolicy:
    """Apply a three-way policy to causally ordered event updates.

    Lower model scores indicate lower expected final risk. SAFE-EXCLUDE is
    available only inside the configured decision window, after
    ``minimum_history`` eligible updates, at or below the calibrated safe
    threshold, and when the optional applicability gate permits it.
    ESCALATE is an operational prioritisation rule, not part of the statistical
    dangerous-exclusion guarantee.
    """

    def __init__(
        self,
        safe_threshold: float,
        minimum_history: int = 1,
        escalation_threshold: float | None = None,
        shift_gate: ConformalShiftGate | None = None,
        min_days_to_tca: float = 2.0,
        max_days_to_tca: float = 7.0,
    ):
        if np.isnan(safe_threshold):
            raise ValueError("safe_threshold must not be NaN")
        if minimum_history < 1:
            raise ValueError("minimum_history must be at least one")
        if not 0 <= min_days_to_tca < max_days_to_tca:
            raise ValueError("Require 0 <= min_days_to_tca < max_days_to_tca")
        if escalation_threshold is not None:
            if not np.isfinite(escalation_threshold):
                raise ValueError("escalation_threshold must be finite when provided")
            if escalation_threshold <= safe_threshold:
                raise ValueError("escalation_threshold must exceed safe_threshold")
        self.safe_threshold = float(safe_threshold)
        self.minimum_history = int(minimum_history)
        self.escalation_threshold = (
            None if escalation_threshold is None else float(escalation_threshold)
        )
        self.shift_gate = shift_gate
        self.min_days_to_tca = float(min_days_to_tca)
        self.max_days_to_tca = float(max_days_to_tca)
        self._counts: dict[Any, int] = {}
        self._eligible_counts: dict[Any, int] = {}
        self._last_tca: dict[Any, float] = {}
        self._audit: list[TriageDecision] = []

    def update(
        self,
        event_id: Any,
        time_to_tca: float,
        score: float,
        gate_features: Mapping[str, Any] | None = None,
    ) -> TriageDecision:
        """Process one newly available CDM score and return an auditable decision."""
        tca = float(time_to_tca)
        value = float(score)
        if not np.isfinite(tca) or tca < 0:
            raise ValueError("time_to_tca must be finite and non-negative")
        if not np.isfinite(value):
            raise ValueError("score must be finite")
        previous_tca = self._last_tca.get(event_id)
        if previous_tca is not None and tca >= previous_tca:
            raise ValueError("Updates for an event must have strictly decreasing time_to_tca")

        sequence_number = self._counts.get(event_id, 0) + 1
        window_eligible = self.min_days_to_tca <= tca <= self.max_days_to_tca
        eligible_history_count = self._eligible_counts.get(event_id, 0)
        if window_eligible:
            eligible_history_count += 1

        gate_allowed = True
        shift_score: float | None = None
        if self.shift_gate is not None:
            if gate_features is None:
                gate_allowed = False
                shift_score = float("inf")
            else:
                frame = pd.DataFrame([dict(gate_features)])
                shift_score = float(self.shift_gate.score(frame).iloc[0])
                gate_allowed = bool(self.shift_gate.allows_safe_exclude(frame).iloc[0])

        if self.escalation_threshold is not None and value >= self.escalation_threshold:
            decision = Decision.ESCALATE
            reason = "score_at_or_above_escalation_threshold"
        elif tca > self.max_days_to_tca:
            decision = Decision.MONITOR
            reason = "decision_window_not_open"
        elif tca < self.min_days_to_tca:
            decision = Decision.MONITOR
            reason = "decision_window_closed"
        elif eligible_history_count < self.minimum_history:
            decision = Decision.MONITOR
            reason = "minimum_history_not_reached"
        elif value <= self.safe_threshold and not gate_allowed:
            decision = Decision.MONITOR
            reason = "safe_exclude_blocked_by_shift_gate"
        elif value <= self.safe_threshold:
            decision = Decision.SAFE_EXCLUDE
            reason = "score_at_or_below_calibrated_threshold"
        else:
            decision = Decision.MONITOR
            reason = "score_between_decision_thresholds"

        result = TriageDecision(
            event_id=event_id,
            sequence_number=sequence_number,
            time_to_tca=tca,
            score=value,
            decision=decision,
            reason=reason,
            shift_score=shift_score,
            shift_gate_allowed=gate_allowed,
            decision_window_eligible=window_eligible,
            eligible_history_count=eligible_history_count,
        )
        self._counts[event_id] = sequence_number
        self._eligible_counts[event_id] = eligible_history_count
        self._last_tca[event_id] = tca
        self._audit.append(result)
        return result

    def audit_log(self) -> pd.DataFrame:
        """Return a copy of all accepted decisions in ingestion order."""
        columns = [field.name for field in TriageDecision.__dataclass_fields__.values()]
        if not self._audit:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame([decision.to_record() for decision in self._audit])

    def reset_event(self, event_id: Any) -> None:
        """Forget runtime state for one event without deleting its audit records."""
        self._counts.pop(event_id, None)
        self._eligible_counts.pop(event_id, None)
        self._last_tca.pop(event_id, None)
