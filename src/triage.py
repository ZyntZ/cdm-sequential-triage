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

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["decision"] = self.decision.value
        return record


class SequentialTriagePolicy:
    """Apply a three-way policy to causally ordered event updates.

    Lower model scores indicate lower expected final risk.  SAFE-EXCLUDE is
    available only after ``minimum_history`` updates, at or below the calibrated
    safe threshold, and when the optional applicability gate permits it.
    ESCALATE is an operational prioritisation rule, not part of the statistical
    dangerous-exclusion guarantee.
    """

    def __init__(
        self,
        safe_threshold: float,
        minimum_history: int = 1,
        escalation_threshold: float | None = None,
        shift_gate: ConformalShiftGate | None = None,
    ):
        if np.isnan(safe_threshold):
            raise ValueError("safe_threshold must not be NaN")
        if minimum_history < 1:
            raise ValueError("minimum_history must be at least one")
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
        self._counts: dict[Any, int] = {}
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
        elif sequence_number < self.minimum_history:
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
        )
        self._counts[event_id] = sequence_number
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
        self._last_tca.pop(event_id, None)
