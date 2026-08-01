"""Parse external CCSDS-like CDM JSON and audit v13 cohort readiness.

The adapter targets the public Space-Track/TraCSS-style flattened JSON field names.
It does not download data or claim that a source is redistributable. Event grouping is
explicit and auditable because public CDM feeds do not expose one universal stable
conjunction-event identifier.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from snapshot_model import CATEGORICAL_FEATURES, NUMERIC_FEATURES

REQUIRED_EXTERNAL_FEATURES = tuple(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
OBJECT_PREFIXES = ("SAT1", "SAT2")
POSITION_COVARIANCE_KEYS = ("CR_R", "CT_R", "CT_T", "CN_R", "CN_T", "CN_N")


def _clean_key(key: Any) -> str:
    return str(key).strip().upper()


def _record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {_clean_key(key): value for key, value in record.items()}


def _number(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _number_with_unit(
    record: Mapping[str, Any],
    field: str,
    accepted_units: set[str],
) -> float:
    value = _number(record.get(field))
    unit = record.get(f"{field}_UNIT")
    if unit not in (None, ""):
        normalized = str(unit).strip().lower().replace(" ", "")
        if normalized not in accepted_units:
            raise ValueError(
                f"{field} uses unsupported unit {unit!r}; expected {sorted(accepted_units)}"
            )
    return value


def _timestamp(value: Any, field: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{field} must contain an ISO-8601 timestamp")
    return parsed


def _source_text(source: str | Path | bytes) -> str:
    if isinstance(source, bytes):
        return source.decode("utf-8-sig")
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8-sig")
    stripped = source.lstrip()
    if (
        "\n" not in source
        and "\r" not in source
        and not stripped.startswith(("{", "["))
    ):
        candidate = Path(source)
        if candidate.exists():
            return candidate.read_text(encoding="utf-8-sig")
    return source


def _kvn_value(value: str) -> tuple[str, str | None]:
    """Split a KVN value from its optional trailing unit declaration."""
    stripped = value.strip()
    if stripped.endswith("]") and "[" in stripped:
        plain, unit = stripped.rsplit("[", 1)
        normalized_unit = unit[:-1].strip()
        if plain.strip() and normalized_unit:
            return plain.strip(), normalized_unit
    return stripped, None


def _parse_cdm_kvn(text: str) -> list[dict[str, Any]]:
    """Parse one or more CCSDS CDM keyword-value notation documents."""
    documents: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    object_prefix: str | None = None

    def finish() -> None:
        nonlocal current, object_prefix
        if current:
            if "CCSDS_CDM_VERS" not in current:
                raise ValueError("KVN document has no CCSDS_CDM_VERS field")
            documents.append(current)
        current = {}
        object_prefix = None

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("COMMENT") or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid KVN line {line_number}: expected KEY = VALUE")
        raw_key, raw_value = line.split("=", 1)
        key = _clean_key(raw_key)
        value, unit = _kvn_value(raw_value)
        if not key or not value:
            raise ValueError(f"Invalid KVN line {line_number}: empty key or value")
        if key == "CCSDS_CDM_VERS" and current:
            finish()
        if key == "OBJECT":
            normalized_object = value.strip().upper().replace("_", "")
            if normalized_object in {"OBJECT1", "SAT1"}:
                object_prefix = "SAT1"
            elif normalized_object in {"OBJECT2", "SAT2"}:
                object_prefix = "SAT2"
            else:
                raise ValueError(
                    f"Invalid KVN line {line_number}: unsupported OBJECT {value!r}"
                )
            continue
        target_key = f"{object_prefix}_{key}" if object_prefix else key
        if target_key in current:
            raise ValueError(f"Duplicate KVN field {target_key!r} on line {line_number}")
        current[target_key] = value
        if unit is not None:
            current[f"{target_key}_UNIT"] = unit
    finish()
    if not documents:
        raise ValueError("CDM KVN input is empty")
    return documents


def parse_cdm_source(
    source: str | Path | bytes | list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Load JSON, newline-delimited JSON, or CCSDS CDM KVN without network access."""
    if isinstance(source, list):
        payload: Any = source
    else:
        text = _source_text(source)
        stripped = text.strip()
        if not stripped:
            raise ValueError("CDM input is empty")
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            try:
                payload = [json.loads(line) for line in stripped.splitlines() if line.strip()]
            except json.JSONDecodeError:
                return [_record(item) for item in _parse_cdm_kvn(stripped)]
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not payload:
        raise ValueError("CDM JSON must contain at least one object")
    if not all(isinstance(item, Mapping) for item in payload):
        raise ValueError("Every CDM JSON item must be an object")
    return [_record(item) for item in payload]


def parse_cdm_json(
    source: str | Path | bytes | list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Backward-compatible alias for :func:`parse_cdm_source`."""
    return parse_cdm_source(source)


def _object_designator(record: Mapping[str, Any], prefix: str) -> str:
    for suffix in ("OBJECT_DESIGNATOR", "NORAD_CAT_ID", "OBJECT_NAME"):
        value = record.get(f"{prefix}_{suffix}")
        if value not in (None, ""):
            return str(value).strip()
    raise ValueError(f"{prefix} has no object designator")


def _pair_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return tuple(sorted((_object_designator(record, "SAT1"), _object_designator(record, "SAT2"))))


def _assign_event_ids(rows: list[dict[str, Any]], tolerance_minutes: int) -> None:
    """Cluster messages by object pair and nearby TCA, recording grouping diagnostics."""
    if tolerance_minutes < 1:
        raise ValueError("tca_tolerance_minutes must be positive")
    tolerance = pd.Timedelta(minutes=tolerance_minutes)
    by_pair: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        by_pair.setdefault(row.pop("__pair"), []).append(index)
    for pair, positions in by_pair.items():
        positions.sort(key=lambda index: pd.Timestamp(rows[index]["tca"]))
        cluster = -1
        previous_tca = None
        for position in positions:
            tca = pd.Timestamp(rows[position]["tca"])
            if previous_tca is None or tca - previous_tca > tolerance:
                cluster += 1
            material = f"{pair[0]}|{pair[1]}|{cluster}".encode("utf-8")
            rows[position]["event_id"] = "ext-" + hashlib.sha256(material).hexdigest()[:24]
            rows[position]["event_pair"] = f"{pair[0]}|{pair[1]}"
            rows[position]["event_cluster"] = cluster
            previous_tca = tca


def _covariance_matrix(record: Mapping[str, Any], prefix: str) -> np.ndarray | None:
    values = [_number(record.get(f"{prefix}_{key}")) for key in POSITION_COVARIANCE_KEYS]
    if not np.isfinite(values).all():
        return None
    rr, tr, tt, nr, nt, nn = values
    matrix = np.array([[rr, tr, nr], [tr, tt, nt], [nr, nt, nn]], dtype=float)
    return matrix


def _positive_determinant(matrix: np.ndarray | None) -> float:
    if matrix is None:
        return float("nan")
    determinant = float(np.linalg.det(matrix))
    return determinant if np.isfinite(determinant) and determinant > 0 else float("nan")


def _mahalanobis(record: Mapping[str, Any], cov1: np.ndarray | None, cov2: np.ndarray | None) -> float:
    supplied = _number(record.get("MAHALANOBIS_DISTANCE"))
    if np.isfinite(supplied):
        return supplied
    relative = np.array([
        _number(record.get("RELATIVE_POSITION_R")),
        _number(record.get("RELATIVE_POSITION_T")),
        _number(record.get("RELATIVE_POSITION_N")),
    ])
    if cov1 is None or cov2 is None or not np.isfinite(relative).all():
        return float("nan")
    try:
        squared = float(relative @ np.linalg.pinv(cov1 + cov2) @ relative)
    except np.linalg.LinAlgError:
        return float("nan")
    return math.sqrt(max(0.0, squared)) if np.isfinite(squared) else float("nan")


def _log_probability(value: Any) -> float:
    probability = _number(value)
    if not np.isfinite(probability) or probability < 0:
        return float("nan")
    return math.log10(probability) if probability > 0 else float("nan")


def _max_risk(record: Mapping[str, Any], risk: float) -> tuple[float, float]:
    maximum = _log_probability(
        record.get("COLLISION_MAX_PROBABILITY", record.get("COLLISION_MAX_PC"))
    )
    scale = _number(
        record.get("COLLISION_MAX_PC_SCALE_FACTOR", record.get("COLLISION_MAX_PC_SCALING"))
    )
    if not np.isfinite(maximum):
        maximum = risk
    return maximum, scale


def adapt_external_cdms(
    records: Iterable[Mapping[str, Any]],
    *,
    tca_tolerance_minutes: int = 30,
) -> pd.DataFrame:
    """Map flattened external CDMs to the frozen ESA-compatible feature contract.

    Missing source fields remain NaN for CatBoost, but readiness reporting exposes
    their prevalence. No labels are created here.
    """
    rows = []
    seen_messages: set[str] = set()
    for raw in records:
        record = _record(raw)
        creation = _timestamp(record.get("CREATION_DATE", record.get("INSERT_EPOCH")), "CREATION_DATE")
        tca = _timestamp(record.get("TCA"), "TCA")
        time_to_tca = (tca - creation).total_seconds() / 86400.0
        if time_to_tca < 0:
            raise ValueError("CDM creation date must not be after TCA")
        message_id = str(record.get("MESSAGE_ID", record.get("CDM_ID", ""))).strip()
        if not message_id:
            message_id = hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()
        if message_id in seen_messages:
            raise ValueError(f"Duplicate CDM message identifier: {message_id}")
        seen_messages.add(message_id)
        risk = _log_probability(record.get("COLLISION_PROBABILITY"))
        maximum, scaling = _max_risk(record, risk)
        cov1, cov2 = _covariance_matrix(record, "SAT1"), _covariance_matrix(record, "SAT2")
        row = {
            "__pair": _pair_key(record),
            "source_message_id": message_id,
            "creation_date": creation.isoformat(),
            "tca": tca.isoformat(),
            "time_to_tca": time_to_tca,
            "risk": risk,
            "max_risk_estimate": maximum,
            "max_risk_scaling": scaling,
            "miss_distance": _number_with_unit(record, "MISS_DISTANCE", {"m", "meter", "metre"}),
            "relative_speed": _number_with_unit(record, "RELATIVE_SPEED", {"m/s", "m/sec", "m/s**1"}),
            "mahalanobis_distance": _mahalanobis(record, cov1, cov2),
            "t_position_covariance_det": _positive_determinant(cov1),
            "c_position_covariance_det": _positive_determinant(cov2),
            "t_obs_available": _number(record.get("SAT1_OBS_AVAILABLE")),
            "t_obs_used": _number(record.get("SAT1_OBS_USED")),
            "t_weighted_rms": _number(record.get("SAT1_WEIGHTED_RMS")),
            "c_obs_available": _number(record.get("SAT2_OBS_AVAILABLE")),
            "c_obs_used": _number(record.get("SAT2_OBS_USED")),
            "c_weighted_rms": _number(record.get("SAT2_WEIGHTED_RMS")),
            "mission_id": _object_designator(record, "SAT1"),
            "c_object_type": str(record.get("SAT2_OBJECT_TYPE", "__MISSING__")),
        }
        rows.append(row)
    _assign_event_ids(rows, tca_tolerance_minutes)
    frame = pd.DataFrame(rows).sort_values(
        ["event_id", "time_to_tca"], ascending=[True, False], kind="mergesort"
    ).reset_index(drop=True)
    if frame.duplicated(["event_id", "time_to_tca"]).any():
        raise ValueError("Event grouping produced duplicate event_id/time_to_tca rows")
    return frame



def outcome_blind_features(
    frame: pd.DataFrame,
    *,
    min_days: float = 2.0,
) -> pd.DataFrame:
    """Return only messages available before the frozen outcome firewall closes."""
    if min_days < 0:
        raise ValueError("min_days must be non-negative")
    selected = frame.loc[frame["time_to_tca"] >= min_days].copy()
    if selected.empty:
        raise ValueError("No CDMs remain before the outcome firewall")
    forbidden = {"y", "final_risk", "provisional_final_risk"}.intersection(selected.columns)
    if forbidden:
        raise ValueError(f"Outcome columns are not allowed in features: {sorted(forbidden)}")
    return selected.reset_index(drop=True)


def derive_event_labels(
    complete_history: pd.DataFrame,
    *,
    collection_complete: bool = False,
    threshold_log10_pc: float = -6.0,
) -> pd.DataFrame:
    """Derive labels only after the caller attests that event collection is complete."""
    if collection_complete is not True:
        raise ValueError("collection_complete=True is required before deriving outcomes")
    required = {"event_id", "time_to_tca", "risk"}
    missing = required.difference(complete_history.columns)
    if missing:
        raise ValueError(f"Missing outcome columns: {sorted(missing)}")
    ordered = complete_history.sort_values(["event_id", "time_to_tca"])
    terminal = ordered.groupby("event_id", as_index=False).first()
    risk = pd.to_numeric(terminal["risk"], errors="coerce")
    if not np.isfinite(risk).all():
        missing_count = int((~np.isfinite(risk)).sum())
        raise ValueError(f"{missing_count} events have no finite terminal collision probability")
    return terminal.loc[:, ["event_id"]].assign(
        y=(risk >= threshold_log10_pc).astype(int).to_numpy()
    )


def event_grouping_review(
    frame: pd.DataFrame,
    *,
    tca_tolerance_minutes: int = 30,
) -> dict[str, Any]:
    """Identify event clusters that need manual review before a study is frozen.

    Public CDM exports do not provide one universal conjunction-event identifier.
    The adapter therefore groups by object pair and TCA proximity. This diagnostic
    makes the remaining ambiguity explicit instead of reporting a blanket warning.
    """
    required = {"event_id", "event_pair", "tca"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing event-grouping columns: {sorted(missing)}")
    if tca_tolerance_minutes < 1:
        raise ValueError("tca_tolerance_minutes must be positive")
    tolerance = pd.Timedelta(minutes=tca_tolerance_minutes)
    clusters = []
    for (pair, event_id), event in frame.groupby(["event_pair", "event_id"], sort=True):
        tca = pd.to_datetime(event["tca"], utc=True, errors="coerce")
        if tca.isna().any():
            raise ValueError(f"Event {event_id!r} contains an invalid TCA")
        clusters.append({
            "event_pair": str(pair),
            "event_id": str(event_id),
            "messages": int(len(event)),
            "tca_min": tca.min(),
            "tca_max": tca.max(),
        })

    flagged: dict[str, dict[str, Any]] = {}
    for cluster in clusters:
        span = cluster["tca_max"] - cluster["tca_min"]
        if span > tolerance:
            flagged[cluster["event_id"]] = {
                "event_id": cluster["event_id"],
                "event_pair": cluster["event_pair"],
                "messages": cluster["messages"],
                "tca_span_minutes": span.total_seconds() / 60.0,
                "nearest_other_event_gap_minutes": None,
                "reasons": ["within_event_tca_span_exceeds_grouping_tolerance"],
            }

    for pair in sorted({cluster["event_pair"] for cluster in clusters}):
        same_pair = sorted(
            (cluster for cluster in clusters if cluster["event_pair"] == pair),
            key=lambda cluster: (cluster["tca_min"], cluster["event_id"]),
        )
        for left, right in zip(same_pair, same_pair[1:]):
            gap = max(pd.Timedelta(0), right["tca_min"] - left["tca_max"])
            if gap <= tolerance:
                gap_minutes = gap.total_seconds() / 60.0
                for cluster in (left, right):
                    item = flagged.setdefault(cluster["event_id"], {
                        "event_id": cluster["event_id"],
                        "event_pair": cluster["event_pair"],
                        "messages": cluster["messages"],
                        "tca_span_minutes": (
                            cluster["tca_max"] - cluster["tca_min"]
                        ).total_seconds() / 60.0,
                        "nearest_other_event_gap_minutes": None,
                        "reasons": [],
                    })
                    previous = item["nearest_other_event_gap_minutes"]
                    if previous is None or gap_minutes < previous:
                        item["nearest_other_event_gap_minutes"] = gap_minutes
                    if "neighboring_same_pair_event_within_tolerance" not in item["reasons"]:
                        item["reasons"].append(
                            "neighboring_same_pair_event_within_tolerance"
                        )

    flagged_events = sorted(flagged.values(), key=lambda item: item["event_id"])
    repeated_pairs = sum(
        len({cluster["event_id"] for cluster in clusters if cluster["event_pair"] == pair}) > 1
        for pair in {cluster["event_pair"] for cluster in clusters}
    )
    return {
        "method": "object pair plus sequential TCA clustering",
        "tca_tolerance_minutes": int(tca_tolerance_minutes),
        "events_reviewed": len(clusters),
        "object_pairs": len({cluster["event_pair"] for cluster in clusters}),
        "object_pairs_with_multiple_events": int(repeated_pairs),
        "flagged_events": len(flagged_events),
        "manual_review_required": bool(flagged_events),
        "flags": flagged_events,
    }

def readiness_report(
    frame: pd.DataFrame,
    *,
    minimum_history: int = 3,
    min_days: float = 2.0,
    max_days: float = 7.0,
    collection_complete: bool = False,
) -> dict[str, Any]:
    """Quantify whether an external collection can enter the frozen v13 study."""
    required = {"event_id", "time_to_tca", "risk", *REQUIRED_EXTERNAL_FEATURES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing normalized columns: {sorted(missing)}")
    window = frame.loc[frame["time_to_tca"].between(min_days, max_days, inclusive="both")]
    history = window.groupby("event_id").size()
    eligible_ids = set(history.loc[history >= minimum_history].index.astype(str))
    all_ids = set(frame["event_id"].astype(str))
    if collection_complete:
        final = (
            frame.sort_values(["event_id", "time_to_tca"])
            .groupby("event_id", as_index=False)
            .nth(0)
            .reset_index(drop=True)
        )
        finite_final_risk = np.isfinite(
            pd.to_numeric(final["risk"], errors="coerce")
        )
        labelled = final.loc[finite_final_risk].copy()
        labelled["y"] = (labelled["risk"].astype(float) >= -6.0).astype(int)
        positive = int(labelled["y"].sum())
        label_fields = {
            "events_without_finite_final_pc": int(len(final) - len(labelled)),
            "provisionally_labelled_events": int(len(labelled)),
            "provisional_positive_events": positive,
            "positive_rate": None if labelled.empty else positive / len(labelled),
            "provisional_calibration_positive_target_met": positive >= 100,
            "provisional_evaluation_positive_target_met": positive >= 200,
            "provisional_total_positive_target_met": positive >= 300,
        }
    else:
        label_fields = {
            "events_without_finite_final_pc": None,
            "provisionally_labelled_events": None,
            "provisional_positive_events": None,
            "positive_rate": None,
            "provisional_calibration_positive_target_met": None,
            "provisional_evaluation_positive_target_met": None,
            "provisional_total_positive_target_met": None,
        }
    missingness = {
        column: float(frame[column].isna().mean())
        for column in REQUIRED_EXTERNAL_FEATURES
    }
    grouping = event_grouping_review(frame)
    return {
        "rows": int(len(frame)),
        "events": len(all_ids),
        "events_with_window_history": int(history.size),
        "events_eligible_minimum_history": len(eligible_ids),
        "collection_complete_attested": bool(collection_complete),
        "positive_counts_suppressed": not bool(collection_complete),
        **label_fields,
        "feature_missing_fraction": missingness,
        "complete_feature_rows": int(frame[list(REQUIRED_EXTERNAL_FEATURES)].notna().all(axis=1).sum()),
        "event_grouping": grouping,
        "scientific_status": (
            "manual-event-grouping-review-required"
            if grouping["manual_review_required"]
            else (
                "candidate-collection-only"
                if len(eligible_ids) > 0
                else "insufficient-sequential-history"
            )
        ),
        "limitations": [
            (
                "terminal Pc counts are available only after collection completeness is attested"
                if not collection_complete
                else "terminal Pc counts are post-completion diagnostics, not confirmation results"
            ),
            "source-to-ESA feature compatibility does not establish exchangeability",
            (
                "flagged event groups require manual review before study freeze"
                if grouping["manual_review_required"]
                else "automated grouping checks cannot exclude all event-identity errors"
            ),
        ],
    }
