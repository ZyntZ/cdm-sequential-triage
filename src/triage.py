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


@dataclass(frozen=True)
class ProcessedBatch:
    scores_sha256: str
    calibration_sha256: str
    rows: int
    events: int
    min_time_to_tca: float
    max_time_to_tca: float
    first_audit_row: int
    last_audit_row: int
    previous_entry_sha256: str | None
    entry_sha256: str

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


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
        self._audit_event_ids: set[Any] = set()
        self._processed_batches: list[ProcessedBatch] = []

    def _decision_for_update(
        self,
        *,
        time_to_tca: float,
        score: float,
        eligible_history_count: int,
        gate_allowed: bool,
    ) -> tuple[Decision, str]:
        """Apply the frozen decision order to validated update fields."""
        if self.escalation_threshold is not None and score >= self.escalation_threshold:
            return Decision.ESCALATE, "score_at_or_above_escalation_threshold"
        if time_to_tca > self.max_days_to_tca:
            return Decision.MONITOR, "decision_window_not_open"
        if time_to_tca < self.min_days_to_tca:
            return Decision.MONITOR, "decision_window_closed"
        if eligible_history_count < self.minimum_history:
            return Decision.MONITOR, "minimum_history_not_reached"
        if score <= self.safe_threshold and not gate_allowed:
            return Decision.MONITOR, "safe_exclude_blocked_by_shift_gate"
        if score <= self.safe_threshold:
            return Decision.SAFE_EXCLUDE, "score_at_or_below_calibrated_threshold"
        return Decision.MONITOR, "score_between_decision_thresholds"


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

        decision, reason = self._decision_for_update(
            time_to_tca=tca,
            score=value,
            eligible_history_count=eligible_history_count,
            gate_allowed=gate_allowed,
        )

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
        self._audit_event_ids.add(event_id)
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

    def processed_batches(self) -> list[dict[str, Any]]:
        """Return a copy of the persisted input-batch provenance ledger."""
        return [entry.to_record() for entry in self._processed_batches]

    def has_processed_batch(self, scores_sha256: str) -> bool:
        return any(
            entry.scores_sha256 == scores_sha256
            for entry in self._processed_batches
        )

    @staticmethod
    def _processed_batch_entry_sha256(record: Mapping[str, Any]) -> str:
        material = {
            key: record[key]
            for key in (
                "scores_sha256", "calibration_sha256", "rows", "events",
                "min_time_to_tca", "max_time_to_tca", "first_audit_row",
                "last_audit_row", "previous_entry_sha256",
            )
        }
        canonical = json.dumps(
            material, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def processed_batch_chain_head(self) -> str | None:
        """Return the verified head hash of the processed-batch ledger."""
        if not self._processed_batches:
            return None
        return self._processed_batches[-1].entry_sha256

    def record_processed_batch(
        self,
        *,
        scores_sha256: str,
        calibration_sha256: str,
        rows: int,
        events: int,
        min_time_to_tca: float,
        max_time_to_tca: float,
        first_audit_row: int,
        last_audit_row: int,
        previous_entry_sha256: str | None = None,
        entry_sha256: str | None = None,
        _require_current_tail: bool = True,
    ) -> None:
        """Append validated batch provenance before checkpoint publication."""
        for name, value in (
            ("scores_sha256", scores_sha256),
            ("calibration_sha256", calibration_sha256),
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a 64-character SHA-256")
            try:
                int(value, 16)
            except ValueError as error:
                raise ValueError(f"{name} must be hexadecimal") from error
        if self.has_processed_batch(scores_sha256):
            raise ValueError("Score batch has already been processed")
        if (
            self._processed_batches
            and calibration_sha256 != self._processed_batches[0].calibration_sha256
        ):
            raise ValueError(
                "All processed batches in a runtime session must use the same calibration"
            )
        rows, events = int(rows), int(events)
        first_audit_row, last_audit_row = int(first_audit_row), int(last_audit_row)
        min_tca, max_tca = float(min_time_to_tca), float(max_time_to_tca)
        if rows < 1 or events < 1 or events > rows:
            raise ValueError("Processed batch row/event counts are invalid")
        if not np.isfinite(min_tca) or not np.isfinite(max_tca) or min_tca > max_tca:
            raise ValueError("Processed batch TCA bounds are invalid")
        if first_audit_row < 1 or last_audit_row - first_audit_row + 1 != rows:
            raise ValueError("Processed batch audit row range is invalid")
        if self._processed_batches:
            expected_first = self._processed_batches[-1].last_audit_row + 1
            if first_audit_row != expected_first:
                raise ValueError("Processed batch audit ranges must be contiguous")
        elif first_audit_row != 1:
            raise ValueError("The first processed batch must start at audit row one")
        if last_audit_row > len(self._audit):
            raise ValueError("Processed batch range exceeds the audit log")
        if _require_current_tail and last_audit_row != len(self._audit):
            raise ValueError("Processed batch must cover the current audit tail")
        expected_previous = (
            None if not self._processed_batches
            else self._processed_batches[-1].entry_sha256
        )
        chain_values_supplied = (
            previous_entry_sha256 is not None or entry_sha256 is not None
        )
        if previous_entry_sha256 is not None:
            if (
                not isinstance(previous_entry_sha256, str)
                or len(previous_entry_sha256) != 64
            ):
                raise ValueError("previous_entry_sha256 must be null or a SHA-256")
            try:
                int(previous_entry_sha256, 16)
            except ValueError as error:
                raise ValueError("previous_entry_sha256 must be hexadecimal") from error
        if chain_values_supplied and previous_entry_sha256 != expected_previous:
            raise ValueError("Processed batch ledger previous hash does not match")
        previous_entry_sha256 = expected_previous
        material = {
            "scores_sha256": scores_sha256,
            "calibration_sha256": calibration_sha256,
            "rows": rows,
            "events": events,
            "min_time_to_tca": min_tca,
            "max_time_to_tca": max_tca,
            "first_audit_row": first_audit_row,
            "last_audit_row": last_audit_row,
            "previous_entry_sha256": expected_previous,
        }
        expected_entry = self._processed_batch_entry_sha256(material)
        if entry_sha256 is not None and entry_sha256 != expected_entry:
            raise ValueError("Processed batch ledger entry hash does not match")
        self._processed_batches.append(ProcessedBatch(
            **material,
            entry_sha256=expected_entry,
        ))

    def reset_event(self, event_id: Any) -> None:
        """Forget untouched event state without invalidating the immutable audit."""
        if event_id in self._audit_event_ids:
            raise RuntimeError(
                "Cannot reset an event after audit records have been written; "
                "use a new event_id for a new conjunction episode"
            )
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
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("Checkpoint float encoding is invalid") from error
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
            "schema_version": 3,
            "configuration": self._configuration_payload(),
            "state": state,
            "audit": audit,
            "processed_batches": [
                entry.to_record() for entry in self._processed_batches
            ],
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
        schema_version = payload.get("schema_version")
        if schema_version not in {1, 2, 3}:
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
            restored_event_id = cls._decode_event_id(record["event_id"])
            policy._audit.append(TriageDecision(
                event_id=restored_event_id,
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
            policy._audit_event_ids.add(restored_event_id)
        audit_tails: dict[Any, TriageDecision] = {}
        for decision in policy._audit:
            previous = audit_tails.get(decision.event_id)
            if previous is None:
                if decision.sequence_number != 1:
                    raise ValueError("Checkpoint audit must start each event at sequence one")
                previous_eligible_count = 0
            else:
                if decision.sequence_number != previous.sequence_number + 1:
                    raise ValueError("Checkpoint audit sequence numbers are not contiguous")
                if decision.time_to_tca >= previous.time_to_tca:
                    raise ValueError("Checkpoint audit TCA is not strictly decreasing")
                previous_eligible_count = previous.eligible_history_count
            expected_window_eligible = (
                policy.min_days_to_tca
                <= decision.time_to_tca
                <= policy.max_days_to_tca
            )
            expected_eligible_count = previous_eligible_count + int(
                expected_window_eligible
            )
            if decision.decision_window_eligible != expected_window_eligible:
                raise ValueError("Checkpoint decision-window flag is inconsistent")
            if decision.eligible_history_count != expected_eligible_count:
                raise ValueError("Checkpoint eligible history is inconsistent")
            if policy.shift_gate is None:
                if decision.shift_score is not None or not decision.shift_gate_allowed:
                    raise ValueError("Checkpoint shift-gate fields are inconsistent")
            expected_decision, expected_reason = policy._decision_for_update(
                time_to_tca=decision.time_to_tca,
                score=decision.score,
                eligible_history_count=decision.eligible_history_count,
                gate_allowed=decision.shift_gate_allowed,
            )
            if decision.decision != expected_decision or decision.reason != expected_reason:
                raise ValueError("Checkpoint decision or reason is inconsistent")
            audit_tails[decision.event_id] = decision
        if set(audit_tails) != set(policy._counts):
            raise ValueError("Checkpoint state and audit event rosters differ")
        for event_id, tail in audit_tails.items():
            if policy._counts[event_id] != tail.sequence_number:
                raise ValueError("Checkpoint state sequence does not match audit tail")
            if policy._eligible_counts[event_id] != tail.eligible_history_count:
                raise ValueError("Checkpoint state history does not match audit tail")
            if policy._last_tca[event_id] != tail.time_to_tca:
                raise ValueError("Checkpoint state TCA does not match audit tail")

        batch_records = [] if schema_version == 1 else payload.get("processed_batches")
        if not isinstance(batch_records, list):
            raise ValueError("Checkpoint processed batch ledger is invalid")
        for record in batch_records:
            if not isinstance(record, dict):
                raise ValueError("Checkpoint processed batch entry is invalid")
            restored_record = dict(record)
            if schema_version < 3:
                restored_record.pop("entry_sha256", None)
                restored_record.pop("previous_entry_sha256", None)
                restored_record["previous_entry_sha256"] = policy.processed_batch_chain_head()
            policy.record_processed_batch(
                **restored_record, _require_current_tail=False
            )
        if batch_records and policy._processed_batches[-1].last_audit_row != len(policy._audit):
            raise ValueError("Checkpoint batch ledger does not cover the audit log")
        return policy
