"""Runtime decision policy for sequential CDM triage.

The predictive model is intentionally kept outside this module.  It supplies a
score after each CDM; this module applies the frozen history, calibration and
applicability rules to that score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import tempfile
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

    def continuation_limits(self) -> dict[Any, float]:
        """Return the last accepted TCA for every active event."""
        return dict(self._last_tca)

    def reset_event(self, event_id: Any) -> None:
        """Forget runtime state for one event without deleting its audit records."""
        self._counts.pop(event_id, None)
        self._eligible_counts.pop(event_id, None)
        self._last_tca.pop(event_id, None)

    @staticmethod
    def _encode_event_id(value: Any) -> dict[str, Any]:
        if isinstance(value, np.generic):
            value = value.item()
        if value is None:
            return {"type": "none", "value": None}
        if isinstance(value, bool):
            return {"type": "bool", "value": value}
        if isinstance(value, int):
            return {"type": "int", "value": value}
        if isinstance(value, float) and np.isfinite(value):
            return {"type": "float", "value": value}
        if isinstance(value, str):
            return {"type": "str", "value": value}
        raise TypeError("event_id must be a JSON-compatible scalar")

    @staticmethod
    def _decode_event_id(payload: Mapping[str, Any]) -> Any:
        kind = payload.get("type")
        value = payload.get("value")
        converters = {
            "none": lambda _: None,
            "bool": bool,
            "int": int,
            "float": float,
            "str": str,
        }
        if kind not in converters:
            raise ValueError(f"Unsupported event_id type: {kind!r}")
        decoded = converters[kind](value)
        if kind == "float" and not np.isfinite(decoded):
            raise ValueError("Checkpoint event_id must be finite")
        return decoded

    @staticmethod
    def _encode_float(value: float | None) -> float | str | None:
        if value is None:
            return None
        numeric = float(value)
        if np.isposinf(numeric):
            return "positive_infinity"
        if np.isneginf(numeric):
            return "negative_infinity"
        if not np.isfinite(numeric):
            raise ValueError("Checkpoint floats must not contain NaN")
        return numeric

    @staticmethod
    def _decode_float(value: float | str | None) -> float | None:
        if value is None:
            return None
        if value == "positive_infinity":
            return float("inf")
        if value == "negative_infinity":
            return float("-inf")
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("Checkpoint floats must be finite or encoded infinity")
        return numeric

    def configuration(self) -> dict[str, Any]:
        """Return the complete JSON-safe runtime configuration."""
        return {
            "safe_threshold": self._encode_float(self.safe_threshold),
            "minimum_history": self.minimum_history,
            "escalation_threshold": self._encode_float(self.escalation_threshold),
            "min_days_to_tca": self.min_days_to_tca,
            "max_days_to_tca": self.max_days_to_tca,
            "shift_gate_fingerprint": (
                None if self.shift_gate is None else self.shift_gate.fingerprint()
            ),
        }

    def configuration_fingerprint(self) -> str:
        """Return a stable SHA-256 fingerprint of the runtime configuration."""
        canonical = json.dumps(
            self.configuration(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _configuration_payload(self) -> dict[str, Any]:
        return self.configuration()

    def checkpoint(self, path: str | Path) -> str:
        """Atomically persist the complete runtime state and return its SHA-256."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        state = []
        for event_id, count in self._counts.items():
            state.append({
                "event_id": self._encode_event_id(event_id),
                "sequence_number": count,
                "eligible_history_count": self._eligible_counts[event_id],
                "last_time_to_tca": self._last_tca[event_id],
            })
        audit = []
        for decision in self._audit:
            record = decision.to_record()
            record["event_id"] = self._encode_event_id(decision.event_id)
            record["shift_score"] = self._encode_float(decision.shift_score)
            audit.append(record)
        payload = {
            "schema_version": 1,
            "configuration": self._configuration_payload(),
            "state": state,
            "audit": audit,
        }
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        envelope = {
            "checkpoint_sha256": digest,
            "payload": payload,
        }
        raw = (json.dumps(
            envelope, indent=2, sort_keys=True, allow_nan=False
        ) + "\n").encode("utf-8")
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=target.parent, prefix=f".{target.name}.",
                suffix=".tmp", delete=False
            ) as stream:
                temporary_name = stream.name
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, target)
            temporary_name = None
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
        return digest

    @classmethod
    def restore(
        cls,
        path: str | Path,
        shift_gate: ConformalShiftGate | None = None,
    ) -> "SequentialTriagePolicy":
        """Restore an atomically written checkpoint after validating its integrity."""
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint payload is missing")
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        if envelope.get("checkpoint_sha256") != digest:
            raise ValueError("Checkpoint integrity validation failed")
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported checkpoint schema version")
        configuration = payload.get("configuration")
        if not isinstance(configuration, dict):
            raise ValueError("Checkpoint configuration is missing")
        supplied_gate_fingerprint = (
            None if shift_gate is None else shift_gate.fingerprint()
        )
        if configuration.get("shift_gate_fingerprint") != supplied_gate_fingerprint:
            raise ValueError("Checkpoint shift gate does not match the supplied gate")
        policy = cls(
            safe_threshold=cls._decode_float(configuration["safe_threshold"]),
            minimum_history=configuration["minimum_history"],
            escalation_threshold=cls._decode_float(configuration["escalation_threshold"]),
            shift_gate=shift_gate,
            min_days_to_tca=configuration["min_days_to_tca"],
            max_days_to_tca=configuration["max_days_to_tca"],
        )
        if policy._configuration_payload() != configuration:
            raise ValueError("Checkpoint policy configuration is inconsistent")

        state_records = payload.get("state")
        audit_records = payload.get("audit")
        if not isinstance(state_records, list) or not isinstance(audit_records, list):
            raise ValueError("Checkpoint state or audit is invalid")
        for record in state_records:
            event_id = cls._decode_event_id(record["event_id"])
            if event_id in policy._counts:
                raise ValueError("Checkpoint contains duplicate event state")
            count = int(record["sequence_number"])
            eligible_count = int(record["eligible_history_count"])
            last_tca = float(record["last_time_to_tca"])
            if count < 1 or not 0 <= eligible_count <= count:
                raise ValueError("Checkpoint event counters are inconsistent")
            if not np.isfinite(last_tca) or last_tca < 0:
                raise ValueError("Checkpoint last_time_to_tca is invalid")
            policy._counts[event_id] = count
            policy._eligible_counts[event_id] = eligible_count
            policy._last_tca[event_id] = last_tca

        for record in audit_records:
            sequence_number = int(record["sequence_number"])
            time_to_tca = float(record["time_to_tca"])
            score = float(record["score"])
            eligible_count = int(record["eligible_history_count"])
            if sequence_number < 1 or not 0 <= eligible_count <= sequence_number:
                raise ValueError("Checkpoint audit counters are inconsistent")
            if not np.isfinite(time_to_tca) or time_to_tca < 0 or not np.isfinite(score):
                raise ValueError("Checkpoint audit contains invalid numeric values")
            policy._audit.append(TriageDecision(
                event_id=cls._decode_event_id(record["event_id"]),
                sequence_number=sequence_number,
                time_to_tca=time_to_tca,
                score=score,
                decision=Decision(record["decision"]),
                reason=str(record["reason"]),
                shift_score=cls._decode_float(record.get("shift_score")),
                shift_gate_allowed=bool(record["shift_gate_allowed"]),
                decision_window_eligible=bool(record["decision_window_eligible"]),
                eligible_history_count=eligible_count,
            ))
        return policy
