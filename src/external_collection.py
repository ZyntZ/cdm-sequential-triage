"""Append-only collection ledger for prospective external CDM exports."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from external_cdm import (
    adapt_external_cdms,
    derive_event_labels,
    outcome_blind_features,
    parse_cdm_json,
    readiness_report,
)
from study import file_sha256

SCHEMA_VERSION = 1


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
            "event_id", "event_cluster", "source_export_sha256", "collection_batch"
        }
    }
    raw = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _new_event_id(pair: str, message_id: str) -> str:
    return "ext-" + hashlib.sha256(f"{pair}|{message_id}".encode()).hexdigest()[:24]


def _batch_entry_sha256(batch: dict[str, Any]) -> str:
    material = {
        key: batch.get(key)
        for key in (
            "batch", "source_sha256", "source_messages", "accepted_messages",
            "duplicate_messages", "artifact_path", "artifact_sha256",
            "first_creation_date", "last_creation_date", "previous_entry_sha256",
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
    return combined


def read_collection(ledger_path: str | Path) -> tuple[dict[str, Any], pd.DataFrame]:
    ledger_path = Path(ledger_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported external collection ledger schema")
    if ledger.get("status") not in {"collecting", "closed"}:
        raise ValueError("Invalid external collection status")
    head = _verify_batch_chain(ledger.get("batches", []))
    if ledger.get("batch_chain_head") != head:
        raise ValueError("External collection batch-chain head is invalid")
    return ledger, _load_batches(ledger, ledger_path)


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



def append_export(
    source_path: str | Path,
    ledger_path: str | Path,
    batches_dir: str | Path,
    *,
    collection_start_utc: str,
    collection_end_utc: str,
    tca_tolerance_minutes: int = 30,
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
    else:
        ledger = {
            "schema_version": SCHEMA_VERSION,
            "status": "collecting",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "tca_tolerance_minutes": int(tca_tolerance_minutes),
            "collection_period": requested_period,
            "batches": [],
            "batch_chain_head": None,
        }
        existing = pd.DataFrame()
    if any(batch["source_sha256"] == source_sha256 for batch in ledger["batches"]):
        raise ValueError("Source export has already been appended")

    records = parse_cdm_json(source_path)
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
    accepted["source_export_sha256"] = source_sha256
    accepted["collection_batch"] = len(ledger["batches"]) + 1

    batches_dir.mkdir(parents=True, exist_ok=True)
    batch_number = len(ledger["batches"]) + 1
    batch_path = batches_dir / f"batch-{batch_number:06d}-{source_sha256[:12]}.parquet"
    if batch_path.exists():
        raise FileExistsError(f"Batch artifact already exists: {batch_path}")
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
        batch_path.unlink(missing_ok=True)
        raise
    return ledger


def materialize_collection(
    ledger_path: str | Path,
    features_output: str | Path,
    readiness_output: str | Path,
) -> dict[str, Any]:
    """Build an outcome-blind snapshot from all verified immutable batches."""
    ledger_path = Path(ledger_path)
    features_output, readiness_output = Path(features_output), Path(readiness_output)
    if features_output.exists() or readiness_output.exists():
        raise FileExistsError("Collection snapshot outputs already exist")
    ledger, complete = read_collection(ledger_path)
    if complete.empty:
        raise ValueError("Collection contains no messages")
    features = outcome_blind_features(complete)
    report = readiness_report(complete)
    report["collection"] = {
        "ledger_path": str(ledger_path.resolve()),
        "ledger_sha256": file_sha256(ledger_path),
        "status": ledger["status"],
        "batches": len(ledger["batches"]),
        "source_sha256s": [batch["source_sha256"] for batch in ledger["batches"]],
    }
    written = []
    try:
        _atomic_parquet(features, features_output)
        written.append(features_output)
        report["features_sha256"] = file_sha256(features_output)
        _atomic_json(report, readiness_output)
        written.append(readiness_output)
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return report


def close_collection(
    ledger_path: str | Path,
    labels_output: str | Path,
) -> dict[str, Any]:
    """Irreversibly close collection and derive terminal event labels once."""
    ledger_path, labels_output = Path(ledger_path), Path(labels_output)
    if labels_output.exists():
        raise FileExistsError(f"Labels output already exists: {labels_output}")
    ledger, complete = read_collection(ledger_path)
    if ledger["status"] != "collecting":
        raise ValueError("External collection is already closed")
    labels = derive_event_labels(complete, collection_complete=True)
    _atomic_parquet(labels, labels_output)
    ledger["status"] = "closed"
    ledger["closed_at_utc"] = datetime.now(timezone.utc).isoformat()
    ledger["labels_path"] = os.path.relpath(labels_output, ledger_path.parent)
    ledger["labels_sha256"] = file_sha256(labels_output)
    ledger["labelled_events"] = int(len(labels))
    ledger["positive_events"] = int(labels["y"].sum())
    try:
        _atomic_json(ledger, ledger_path)
    except Exception:
        labels_output.unlink(missing_ok=True)
        raise
    return ledger
