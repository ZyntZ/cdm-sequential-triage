"""Append-only collection ledger for prospective external CDM exports."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None

from external_cdm import (
    adapt_external_cdms,
    derive_event_labels,
    outcome_blind_features,
    parse_cdm_source,
    readiness_report,
)
from study import file_sha256, read_locked_study

SCHEMA_VERSION = 2


@contextmanager
def ledger_lock(ledger_path: str | Path):
    """Serialize ledger mutations with an OS advisory lock."""
    if fcntl is None and msvcrt is None:
        raise RuntimeError("External collection locking is unavailable on this platform")
    ledger_path = Path(ledger_path)
    lock_path = ledger_path.with_suffix(ledger_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        if fcntl is not None:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        else:
            stream.seek(0)
            if stream.read(1) == b"":
                stream.write(b"\0")
                stream.flush()
                os.fsync(stream.fileno())
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield lock_path
        finally:
            if fcntl is not None:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            else:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    raw = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
        frame.to_parquet(temporary, index=False)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _canonical_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def message_fingerprint(row: pd.Series | dict[str, Any]) -> str:
    record = dict(row)
    material = {
        key: _canonical_value(value)
        for key, value in sorted(record.items())
        if key not in {
            "event_id", "event_cluster", "study_cohort",
            "source_export_sha256", "collection_batch"
        }
    }
    raw = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _new_event_id(pair: str, message_id: str) -> str:
    return "ext-" + hashlib.sha256(f"{pair}|{message_id}".encode()).hexdigest()[:24]


def _logical_batch_fingerprints(frame: pd.DataFrame) -> dict[str, str]:
    """Return one canonical content fingerprint per source message."""
    if "source_message_id" not in frame.columns:
        raise ValueError("Batch artifact has no source_message_id column")
    message_ids = frame["source_message_id"].astype(str)
    if message_ids.duplicated().any():
        raise ValueError("Batch artifact contains duplicate source_message_id values")
    return {
        str(row.source_message_id): message_fingerprint(row._asdict())
        for row in frame.itertuples(index=False)
    }


def _recoverable_orphan_matches(path: Path, expected: pd.DataFrame) -> bool:
    """Verify an unregistered batch artifact against re-derived logical rows."""
    try:
        orphan = pd.read_parquet(path)
    except Exception as error:
        raise FileExistsError(
            f"Unregistered batch artifact is unreadable: {path}"
        ) from error
    try:
        orphan_fingerprints = _logical_batch_fingerprints(orphan)
        expected_fingerprints = _logical_batch_fingerprints(expected)
    except ValueError as error:
        raise FileExistsError(
            f"Unregistered batch artifact cannot be verified: {path}"
        ) from error
    return orphan_fingerprints == expected_fingerprints


def _batch_entry_sha256(batch: dict[str, Any]) -> str:
    material = {
        key: batch.get(key)
        for key in (
            "batch", "source_sha256", "source_messages", "accepted_messages",
            "duplicate_messages", "artifact_path", "artifact_sha256",
            "first_creation_date", "last_creation_date", "previous_entry_sha256",
            "recovered_orphan_batch",
        )
    }
    raw = json.dumps(
        material, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _verify_batch_chain(batches: list[dict[str, Any]]) -> str | None:
    previous = None
    for expected_number, batch in enumerate(batches, start=1):
        if int(batch.get("batch", -1)) != expected_number:
            raise ValueError("External collection batch numbers are not contiguous")
        if batch.get("previous_entry_sha256") != previous:
            raise ValueError("External collection batch chain link is invalid")
        expected = _batch_entry_sha256(batch)
        if batch.get("entry_sha256") != expected:
            raise ValueError("External collection batch entry digest is invalid")
        previous = expected
    return previous



def allocate_prospective_cohort(
    event_id: str,
    seed: int,
    calibration_fraction: float,
) -> str:
    """Assign an event without labels using a frozen SHA-256 rule."""
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must lie in (0, 1)")
    digest = hashlib.sha256(f"{int(seed)}:{event_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "calibration" if value < calibration_fraction else "evaluation"


def _assignment_sha256(assignments: dict[str, str]) -> str:
    raw = json.dumps(
        assignments, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _verify_assignments(ledger: dict[str, Any]) -> None:
    allocation = ledger.get("allocation")
    if allocation is None:
        if ledger.get("schema_version") == 1:
            return
        raise ValueError("Prospective collection has no cohort allocation")
    if allocation.get("rule") != "sha256-event-id":
        raise ValueError("Unsupported prospective allocation rule")
    fraction = float(allocation.get("calibration_fraction"))
    seed = int(allocation.get("seed"))
    assignments = allocation.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("Prospective cohort assignments are missing")
    if any(value not in {"calibration", "evaluation"} for value in assignments.values()):
        raise ValueError("Prospective cohort assignment contains an invalid cohort")
    expected = {
        event_id: allocate_prospective_cohort(event_id, seed, fraction)
        for event_id in assignments
    }
    if assignments != expected:
        raise ValueError("Prospective cohort assignment does not match its frozen rule")
    if allocation.get("assignments_sha256") != _assignment_sha256(assignments):
        raise ValueError("Prospective cohort assignment digest is invalid")


def _load_batches(ledger: dict[str, Any], ledger_path: Path) -> pd.DataFrame:
    frames = []
    for batch in ledger.get("batches", []):
        path = Path(batch["artifact_path"])
        if not path.is_absolute():
            path = ledger_path.parent / path
        if file_sha256(path) != batch["artifact_sha256"]:
            raise ValueError(f"Collected batch does not match ledger: {path}")
        frame = pd.read_parquet(path)
        if len(frame) != int(batch["accepted_messages"]):
            raise ValueError(f"Collected batch row count does not match ledger: {path}")
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if combined["source_message_id"].duplicated().any():
        raise ValueError("Collection ledger contains duplicate source_message_id values")
    allocation = ledger.get("allocation")
    if allocation is not None:
        assignments = allocation["assignments"]
        actual_events = set(combined["event_id"].astype(str))
        if actual_events != set(assignments):
            raise ValueError("Collected event roster does not match cohort assignments")
        expected_cohort = combined["event_id"].astype(str).map(assignments)
        if "study_cohort" not in combined or not combined["study_cohort"].eq(expected_cohort).all():
            raise ValueError("Collected messages do not match frozen cohort assignments")
    return combined


def read_collection(ledger_path: str | Path) -> tuple[dict[str, Any], pd.DataFrame]:
    ledger_path = Path(ledger_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("schema_version") not in {1, SCHEMA_VERSION}:
        raise ValueError("Unsupported external collection ledger schema")
    if ledger.get("status") not in {"collecting", "sealed", "closed"}:
        raise ValueError("Invalid external collection status")
    _verify_assignments(ledger)
    head = _verify_batch_chain(ledger.get("batches", []))
    if ledger.get("batch_chain_head") != head:
        raise ValueError("External collection batch-chain head is invalid")
    return ledger, _load_batches(ledger, ledger_path)


def collection_status(
    ledger_path: str | Path,
    *,
    minimum_history: int = 3,
    min_days: float = 2.0,
    max_days: float = 7.0,
) -> dict[str, Any]:
    """Return an integrity-checked, outcome-blind collection progress summary."""
    if minimum_history < 1:
        raise ValueError("minimum_history must be at least one")
    if not np.isfinite(min_days) or not np.isfinite(max_days) or min_days > max_days:
        raise ValueError("Require finite min_days <= max_days")

    ledger_path = Path(ledger_path)
    ledger, complete = read_collection(ledger_path)
    allocation = ledger.get("allocation")
    assignments = {} if allocation is None else allocation["assignments"]
    period = ledger.get("collection_period")
    period_ended = None
    if period is not None:
        period_ended = bool(pd.Timestamp.now(tz="UTC") >= pd.Timestamp(period["end_utc"]))

    if complete.empty:
        history = pd.Series(dtype="int64")
        feature_events: set[str] = set()
    else:
        window = complete.loc[
            complete["time_to_tca"].between(min_days, max_days, inclusive="both")
        ]
        history = window.groupby("event_id").size()
        feature_events = set(window["event_id"].astype(str))
    eligible_events = set(history.loc[history >= minimum_history].index.astype(str))

    cohort_rows: dict[str, dict[str, int]] = {}
    for cohort in ("calibration", "evaluation"):
        event_ids = {
            event_id for event_id, assigned in assignments.items()
            if assigned == cohort
        }
        if complete.empty:
            messages = 0
        else:
            messages = int(complete["event_id"].astype(str).isin(event_ids).sum())
        cohort_rows[cohort] = {
            "assigned_events": len(event_ids),
            "messages": messages,
            "events_in_decision_window": len(event_ids & feature_events),
            "events_eligible_minimum_history": len(event_ids & eligible_events),
        }

    return {
        "schema_version": 1,
        "status": ledger["status"],
        "integrity_verified": True,
        "ledger_sha256": file_sha256(ledger_path),
        "batch_chain_head": ledger.get("batch_chain_head"),
        "batches": len(ledger.get("batches", [])),
        "messages": int(len(complete)),
        "events": int(complete["event_id"].nunique()) if not complete.empty else 0,
        "collection_period": period,
        "collection_period_ended": period_ended,
        "decision_window_days": {"minimum": float(min_days), "maximum": float(max_days)},
        "minimum_history": int(minimum_history),
        "events_in_decision_window": len(feature_events),
        "events_eligible_minimum_history": len(eligible_events),
        "allocation": None if allocation is None else {
            "rule": allocation["rule"],
            "seed": int(allocation["seed"]),
            "calibration_fraction": float(allocation["calibration_fraction"]),
            "assignments_sha256": allocation["assignments_sha256"],
            "cohorts": cohort_rows,
        },
        "outcomes_accessed": ledger["status"] == "closed",
    }


def _assign_persistent_events(
    incoming: pd.DataFrame,
    existing: pd.DataFrame,
    tolerance_minutes: int,
) -> pd.DataFrame:
    tolerance = pd.Timedelta(minutes=tolerance_minutes)
    result = incoming.copy()
    existing_events: dict[str, list[dict[str, Any]]] = {}
    if not existing.empty:
        for event_id, event in existing.groupby("event_id", sort=False):
            pairs = event["event_pair"].astype(str).unique()
            if len(pairs) != 1:
                raise ValueError(f"Existing event {event_id!r} has multiple object pairs")
            existing_events.setdefault(pairs[0], []).append({
                "event_id": str(event_id),
                "tca_min": pd.to_datetime(event["tca"], utc=True).min(),
                "tca_max": pd.to_datetime(event["tca"], utc=True).max(),
            })
    assignments: list[str] = []
    for row in result.itertuples(index=False):
        pair = str(row.event_pair)
        tca = pd.Timestamp(row.tca)
        candidates = []
        for event in existing_events.get(pair, []):
            distance = min(abs(tca - event["tca_min"]), abs(tca - event["tca_max"]))
            if distance <= tolerance:
                candidates.append((distance, event))
        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous event assignment for message {row.source_message_id!r}; "
                f"{len(candidates)} existing events are within the TCA tolerance"
            )
        if candidates:
            event = candidates[0][1]
            event_id = event["event_id"]
            event["tca_min"] = min(event["tca_min"], tca)
            event["tca_max"] = max(event["tca_max"], tca)
        else:
            event_id = _new_event_id(pair, str(row.source_message_id))
            existing_events.setdefault(pair, []).append({
                "event_id": event_id, "tca_min": tca, "tca_max": tca,
            })
        assignments.append(event_id)
    result["event_id"] = assignments
    return result


def _collection_period(start_utc: str, end_utc: str) -> dict[str, str]:
    start = pd.to_datetime(start_utc, utc=True, errors="coerce")
    end = pd.to_datetime(end_utc, utc=True, errors="coerce")
    if pd.isna(start) or pd.isna(end) or start >= end:
        raise ValueError("Require valid collection_start_utc < collection_end_utc")
    return {"start_utc": start.isoformat(), "end_utc": end.isoformat()}


def _validate_collection_period(frame: pd.DataFrame, period: dict[str, str]) -> None:
    creation = pd.to_datetime(frame["creation_date"], utc=True, errors="coerce")
    if creation.isna().any():
        raise ValueError("Collected messages contain invalid creation dates")
    start, end = pd.Timestamp(period["start_utc"]), pd.Timestamp(period["end_utc"])
    outside = ~creation.between(start, end, inclusive="both")
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} CDM messages fall outside the frozen collection period"
        )



def _append_export_unlocked(
    source_path: str | Path,
    ledger_path: str | Path,
    batches_dir: str | Path,
    *,
    collection_start_utc: str,
    collection_end_utc: str,
    tca_tolerance_minutes: int = 30,
    allocation_seed: int = 24072026,
    calibration_fraction: float = 1 / 3,
) -> dict[str, Any]:
    """Append one immutable raw export to a collection without opening outcomes."""
    source_path, ledger_path, batches_dir = map(Path, (source_path, ledger_path, batches_dir))
    source_sha256 = file_sha256(source_path)
    requested_period = _collection_period(collection_start_utc, collection_end_utc)
    if ledger_path.exists():
        ledger, existing = read_collection(ledger_path)
        if ledger["status"] != "collecting":
            raise ValueError("External collection is closed")
        if int(ledger["tca_tolerance_minutes"]) != int(tca_tolerance_minutes):
            raise ValueError("TCA tolerance cannot change after collection starts")
        if ledger.get("collection_period") != requested_period:
            raise ValueError("Collection period cannot change after collection starts")
        if ledger.get("schema_version") == 1:
            raise ValueError("Legacy collection cannot be extended with prospective allocation")
        allocation = ledger["allocation"]
        if int(allocation["seed"]) != int(allocation_seed):
            raise ValueError("Allocation seed cannot change after collection starts")
        if float(allocation["calibration_fraction"]) != float(calibration_fraction):
            raise ValueError("Calibration fraction cannot change after collection starts")
    else:
        if not 0 < float(calibration_fraction) < 1:
            raise ValueError("calibration_fraction must lie in (0, 1)")
        ledger = {
            "schema_version": SCHEMA_VERSION,
            "status": "collecting",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "tca_tolerance_minutes": int(tca_tolerance_minutes),
            "collection_period": requested_period,
            "batches": [],
            "batch_chain_head": None,
            "allocation": {
                "rule": "sha256-event-id",
                "seed": int(allocation_seed),
                "calibration_fraction": float(calibration_fraction),
                "assigned_before_outcome_access": True,
                "assignments": {},
                "assignments_sha256": _assignment_sha256({}),
            },
        }
        existing = pd.DataFrame()
    if any(batch["source_sha256"] == source_sha256 for batch in ledger["batches"]):
        raise ValueError("Source export has already been appended")

    records = parse_cdm_source(source_path)
    incoming = adapt_external_cdms(records, tca_tolerance_minutes=tca_tolerance_minutes)
    _validate_collection_period(incoming, requested_period)
    existing_by_id = {}
    if not existing.empty:
        existing_by_id = {
            str(row.source_message_id): message_fingerprint(row._asdict())
            for row in existing.itertuples(index=False)
        }
    accepted_rows = []
    duplicate_messages = 0
    for row in incoming.itertuples(index=False):
        record = row._asdict()
        message_id = str(record["source_message_id"])
        fingerprint = message_fingerprint(record)
        if message_id in existing_by_id:
            if existing_by_id[message_id] != fingerprint:
                raise ValueError(f"Conflicting re-export of source message {message_id!r}")
            duplicate_messages += 1
        else:
            accepted_rows.append(record)
    if not accepted_rows:
        raise ValueError("Source export contains no new CDM messages")
    accepted = pd.DataFrame(accepted_rows)
    accepted = _assign_persistent_events(accepted, existing, tca_tolerance_minutes)
    assignments = ledger["allocation"]["assignments"]
    for event_id in sorted(accepted["event_id"].astype(str).unique()):
        cohort = allocate_prospective_cohort(
            event_id, allocation_seed, calibration_fraction
        )
        if event_id in assignments and assignments[event_id] != cohort:
            raise ValueError("Existing prospective cohort assignment changed")
        assignments[event_id] = cohort
    ledger["allocation"]["assignments_sha256"] = _assignment_sha256(assignments)
    accepted["study_cohort"] = accepted["event_id"].astype(str).map(assignments)
    accepted["source_export_sha256"] = source_sha256
    accepted["collection_batch"] = len(ledger["batches"]) + 1

    batches_dir.mkdir(parents=True, exist_ok=True)
    batch_number = len(ledger["batches"]) + 1
    batch_path = batches_dir / f"batch-{batch_number:06d}-{source_sha256[:12]}.parquet"
    recovered_orphan = False
    if batch_path.exists():
        relative_batch_path = os.path.relpath(batch_path, ledger_path.parent)
        registered_paths = {
            str(batch.get("artifact_path")) for batch in ledger.get("batches", [])
        }
        if relative_batch_path in registered_paths:
            raise FileExistsError(f"Registered batch artifact already exists: {batch_path}")
        if not _recoverable_orphan_matches(batch_path, accepted):
            raise FileExistsError(
                f"Unregistered batch artifact differs from expected content: {batch_path}"
            )
        recovered_orphan = True
    else:
        _atomic_parquet(accepted, batch_path)
    batch_record = {
        "batch": batch_number,
        "appended_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_path.resolve()),
        "source_sha256": source_sha256,
        "source_messages": len(records),
        "accepted_messages": int(len(accepted)),
        "duplicate_messages": duplicate_messages,
        "artifact_path": os.path.relpath(batch_path, ledger_path.parent),
        "artifact_sha256": file_sha256(batch_path),
        "first_creation_date": str(accepted["creation_date"].min()),
        "last_creation_date": str(accepted["creation_date"].max()),
        "previous_entry_sha256": ledger.get("batch_chain_head"),
        "recovered_orphan_batch": recovered_orphan,
    }
    batch_record["entry_sha256"] = _batch_entry_sha256(batch_record)
    ledger["batches"].append(batch_record)
    ledger["batch_chain_head"] = batch_record["entry_sha256"]
    ledger["messages"] = int(len(existing) + len(accepted))
    ledger["events"] = int(pd.concat([existing, accepted], ignore_index=True)["event_id"].nunique())
    ledger["updated_at_utc"] = batch_record["appended_at_utc"]
    try:
        _atomic_json(ledger, ledger_path)
    except Exception:
        if not recovered_orphan:
            batch_path.unlink(missing_ok=True)
        raise
    return ledger


def append_export(
    source_path: str | Path,
    ledger_path: str | Path,
    batches_dir: str | Path,
    **options,
) -> dict[str, Any]:
    """Append one export while holding the collection ledger lock."""
    with ledger_lock(ledger_path):
        return _append_export_unlocked(
            source_path, ledger_path, batches_dir, **options
        )


def materialize_collection(
    ledger_path: str | Path,
    features_output: str | Path,
    readiness_output: str | Path,
    *,
    calibration_features: str | Path | None = None,
    evaluation_features: str | Path | None = None,
    calibration_roster: str | Path | None = None,
    evaluation_roster: str | Path | None = None,
    allocation_output: str | Path | None = None,
) -> dict[str, Any]:
    """Build an outcome-blind snapshot from all verified immutable batches."""
    ledger_path = Path(ledger_path)
    features_output, readiness_output = Path(features_output), Path(readiness_output)
    cohort_options = (
        calibration_features, evaluation_features, calibration_roster,
        evaluation_roster, allocation_output,
    )
    if any(value is not None for value in cohort_options) and not all(
        value is not None for value in cohort_options
    ):
        raise ValueError(
            "Cohort materialization requires calibration/evaluation features, "
            "rosters, and allocation output together"
        )
    cohort_paths = [Path(value) for value in cohort_options if value is not None]
    all_outputs = [features_output, readiness_output, *cohort_paths]
    existing_outputs = [str(path) for path in all_outputs if path.exists()]
    if existing_outputs:
        raise FileExistsError(
            f"Collection snapshot outputs already exist: {existing_outputs}"
        )
    ledger, complete = read_collection(ledger_path)
    if complete.empty:
        raise ValueError("Collection contains no messages")
    if cohort_paths and ledger["status"] != "sealed":
        raise ValueError(
            "Prospective cohort artifacts require a sealed outcome-blind collection"
        )
    features = outcome_blind_features(complete)
    report = readiness_report(complete, collection_complete=False)
    grouping = report["event_grouping"]
    if cohort_paths and grouping["manual_review_required"]:
        flagged = ", ".join(
            item["event_id"] for item in grouping["flags"][:10]
        )
        suffix = "" if grouping["flagged_events"] <= 10 else ", ..."
        raise ValueError(
            "Prospective cohort materialization is blocked by ambiguous event "
            f"grouping ({grouping['flagged_events']} events: {flagged}{suffix})"
        )
    report["collection"] = {
        "ledger_path": str(ledger_path.resolve()),
        "ledger_sha256": file_sha256(ledger_path),
        "status": ledger["status"],
        "batches": len(ledger["batches"]),
        "source_sha256s": [batch["source_sha256"] for batch in ledger["batches"]],
        "allocation_seed": None if ledger.get("allocation") is None else ledger["allocation"]["seed"],
        "calibration_fraction": None if ledger.get("allocation") is None else ledger["allocation"]["calibration_fraction"],
        "assignments_sha256": None if ledger.get("allocation") is None else ledger["allocation"]["assignments_sha256"],
    }
    written: list[Path] = []
    try:
        _atomic_parquet(features, features_output)
        written.append(features_output)
        report["features_sha256"] = file_sha256(features_output)

        if cohort_paths:
            assignments = ledger["allocation"]["assignments"]
            calibration_ids = sorted(
                event_id for event_id, cohort in assignments.items()
                if cohort == "calibration"
            )
            evaluation_ids = sorted(
                event_id for event_id, cohort in assignments.items()
                if cohort == "evaluation"
            )
            if not calibration_ids or not evaluation_ids:
                raise ValueError(
                    "Prospective allocation must contain both calibration and evaluation events"
                )
            calibration_frame = features.loc[
                features["event_id"].astype(str).isin(calibration_ids)
            ].copy()
            evaluation_frame = features.loc[
                features["event_id"].astype(str).isin(evaluation_ids)
            ].copy()
            if calibration_frame.empty or evaluation_frame.empty:
                raise ValueError(
                    "Both prospective cohorts require at least one outcome-blind feature row"
                )
            calibration_roster_frame = pd.DataFrame({"event_id": calibration_ids})
            evaluation_roster_frame = pd.DataFrame({"event_id": evaluation_ids})
            output_frames = (
                (calibration_frame, Path(calibration_features)),
                (evaluation_frame, Path(evaluation_features)),
                (calibration_roster_frame, Path(calibration_roster)),
                (evaluation_roster_frame, Path(evaluation_roster)),
            )
            for frame, path in output_frames:
                _atomic_parquet(frame, path)
                written.append(path)
            allocation_artifact = {
                "schema_version": 1,
                "status": "assigned-before-outcome-access",
                "outcomes_accessed": False,
                "ledger_path": str(ledger_path.resolve()),
                "ledger_sha256": file_sha256(ledger_path),
                "collection_status": ledger["status"],
                "rule": ledger["allocation"]["rule"],
                "seed": ledger["allocation"]["seed"],
                "calibration_fraction": ledger["allocation"]["calibration_fraction"],
                "assignments_sha256": ledger["allocation"]["assignments_sha256"],
                "events": len(assignments),
                "calibration_events": len(calibration_ids),
                "evaluation_events": len(evaluation_ids),
                "calibration_feature_events": int(
                    calibration_frame["event_id"].nunique()
                ),
                "evaluation_feature_events": int(
                    evaluation_frame["event_id"].nunique()
                ),
                "outputs": {
                    "calibration_features": {
                        "path": str(Path(calibration_features).resolve()),
                        "sha256": file_sha256(calibration_features),
                    },
                    "evaluation_features": {
                        "path": str(Path(evaluation_features).resolve()),
                        "sha256": file_sha256(evaluation_features),
                    },
                    "calibration_roster": {
                        "path": str(Path(calibration_roster).resolve()),
                        "sha256": file_sha256(calibration_roster),
                    },
                    "evaluation_roster": {
                        "path": str(Path(evaluation_roster).resolve()),
                        "sha256": file_sha256(evaluation_roster),
                    },
                },
            }
            _atomic_json(allocation_artifact, Path(allocation_output))
            written.append(Path(allocation_output))
            report["prospective_allocation"] = allocation_artifact

        _atomic_json(report, readiness_output)
        written.append(readiness_output)
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return report


def _seal_collection_unlocked(ledger_path: str | Path) -> dict[str, Any]:
    """Stop ingestion without deriving or accessing terminal labels."""
    ledger_path = Path(ledger_path)
    ledger, complete = read_collection(ledger_path)
    if ledger["status"] != "collecting":
        raise ValueError("Only a collecting external collection can be sealed")
    if complete.empty:
        raise ValueError("Cannot seal an empty external collection")
    period_end = pd.Timestamp(ledger["collection_period"]["end_utc"])
    now = pd.Timestamp.now(tz="UTC")
    if now < period_end:
        raise ValueError(
            "Collection period has not ended; early sealing would permit outcome-dependent selection"
        )
    ledger["status"] = "sealed"
    ledger["sealed_at_utc"] = datetime.now(timezone.utc).isoformat()
    ledger["sealed_messages"] = int(len(complete))
    ledger["sealed_events"] = int(complete["event_id"].nunique())
    ledger["sealed_assignments_sha256"] = ledger["allocation"]["assignments_sha256"]
    _atomic_json(ledger, ledger_path)
    return ledger



def seal_collection(ledger_path: str | Path) -> dict[str, Any]:
    """Seal one collection while holding the collection ledger lock."""
    with ledger_lock(ledger_path):
        return _seal_collection_unlocked(ledger_path)


def _label_artifact_matches(path: Path, expected: pd.DataFrame) -> bool:
    """Return whether an interrupted label output exactly matches re-derived labels."""
    try:
        actual = pd.read_parquet(path)
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=True,
            check_like=False,
        )
    except (OSError, ValueError, TypeError, AssertionError):
        return False
    return True


def _close_collection_unlocked(
    ledger_path: str | Path,
    labels_output: str | Path,
    *,
    study_manifest: str | Path,
    study_lock: str | Path,
    calibration_labels_output: str | Path,
    evaluation_labels_output: str | Path,
) -> dict[str, Any]:
    """Open outcomes once and publish labels only through frozen cohort rosters."""
    ledger_path, labels_output = Path(ledger_path), Path(labels_output)
    calibration_labels_output = Path(calibration_labels_output)
    evaluation_labels_output = Path(evaluation_labels_output)
    label_outputs = [
        labels_output, calibration_labels_output, evaluation_labels_output,
    ]
    ledger, complete = read_collection(ledger_path)
    if ledger["status"] != "sealed":
        raise ValueError(
            "External collection must be sealed and its cohorts frozen before labels are derived"
        )
    if ledger.get("sealed_assignments_sha256") != ledger["allocation"]["assignments_sha256"]:
        raise ValueError("Cohort assignments changed after collection sealing")
    study, study_sha256 = read_locked_study(study_manifest, study_lock)
    allocation = study.get("allocation")
    if not isinstance(allocation, dict):
        raise ValueError("Frozen study is not bound to a prospective allocation")
    if allocation.get("collection_status") != "sealed":
        raise ValueError("Frozen study allocation was not created from a sealed collection")
    current_ledger_sha256 = file_sha256(ledger_path)
    if allocation.get("ledger_sha256") != current_ledger_sha256:
        raise ValueError("Frozen study does not match the sealed collection ledger")
    if allocation.get("assignments_sha256") != ledger["allocation"]["assignments_sha256"]:
        raise ValueError("Frozen study does not match the prospective cohort assignments")
    labels = derive_event_labels(complete, collection_complete=True)
    calibration_ids = set(study["cohorts"]["calibration"]["event_ids"])
    evaluation_ids = set(study["cohorts"]["evaluation"]["event_ids"])
    labelled_ids = set(labels["event_id"].astype(str))
    if calibration_ids.intersection(evaluation_ids):
        raise ValueError("Frozen study cohorts overlap")
    if calibration_ids.union(evaluation_ids) != labelled_ids:
        missing = len(labelled_ids.difference(calibration_ids.union(evaluation_ids)))
        extra = len(calibration_ids.union(evaluation_ids).difference(labelled_ids))
        raise ValueError(
            f"Frozen study label denominator mismatch: {missing} unassigned and {extra} unknown events"
        )
    calibration_labels = labels.loc[
        labels["event_id"].astype(str).isin(calibration_ids)
    ].copy()
    evaluation_labels = labels.loc[
        labels["event_id"].astype(str).isin(evaluation_ids)
    ].copy()
    label_frames = (
        (labels, labels_output),
        (calibration_labels, calibration_labels_output),
        (evaluation_labels, evaluation_labels_output),
    )
    recovered: list[Path] = []
    for frame, path in label_frames:
        if path.exists():
            if not _label_artifact_matches(path, frame):
                raise FileExistsError(
                    f"Existing label output does not match re-derived labels: {path}"
                )
            recovered.append(path)

    written: list[Path] = []
    try:
        for frame, path in label_frames:
            if path in recovered:
                continue
            _atomic_parquet(frame, path)
            written.append(path)
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    ledger["status"] = "closed"
    ledger["closed_at_utc"] = datetime.now(timezone.utc).isoformat()
    ledger["labels_path"] = os.path.relpath(labels_output, ledger_path.parent)
    ledger["labels_sha256"] = file_sha256(labels_output)
    ledger["calibration_labels_path"] = os.path.relpath(
        calibration_labels_output, ledger_path.parent
    )
    ledger["calibration_labels_sha256"] = file_sha256(
        calibration_labels_output
    )
    ledger["evaluation_labels_path"] = os.path.relpath(
        evaluation_labels_output, ledger_path.parent
    )
    ledger["evaluation_labels_sha256"] = file_sha256(
        evaluation_labels_output
    )
    ledger["calibration_events"] = int(len(calibration_labels))
    ledger["evaluation_events"] = int(len(evaluation_labels))
    ledger["study_manifest_path"] = str(Path(study_manifest).resolve())
    ledger["study_manifest_sha256"] = study_sha256
    ledger["study_lock_sha256"] = file_sha256(study_lock)
    ledger["labelled_events"] = int(len(labels))
    ledger["positive_events"] = int(labels["y"].sum())
    try:
        _atomic_json(ledger, ledger_path)
    except BaseException:
        for path in label_outputs:
            path.unlink(missing_ok=True)
        raise
    return ledger

def close_collection(
    ledger_path: str | Path,
    labels_output: str | Path,
    **options,
) -> dict[str, Any]:
    """Close one collection while holding the collection ledger lock."""
    with ledger_lock(ledger_path):
        return _close_collection_unlocked(
            ledger_path, labels_output, **options
        )

