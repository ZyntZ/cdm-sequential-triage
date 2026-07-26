"""Reproducible event-level partitions for the ESA CDM training data."""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED_STAGE1 = 24072026
SEED_STAGE2 = 24072027
FINAL_RISK_THRESHOLD = -6.0
EXPECTED_ROWS = 162634
EXPECTED_EVENTS = 13154
EXPECTED_POSITIVES = 365


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_training_archive(path: str | Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith("train_data.csv")]
        if len(members) != 1:
            raise ValueError("Archive must contain exactly one train_data.csv")
        with archive.open(members[0]) as stream:
            frame = pd.read_csv(stream)
    if len(frame) != EXPECTED_ROWS or frame["event_id"].nunique() != EXPECTED_EVENTS:
        raise ValueError("Unexpected ESA training dataset dimensions")
    return frame


def event_labels(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"event_id", "time_to_tca", "risk"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.duplicated(["event_id", "time_to_tca"]).any():
        raise ValueError("Duplicate event_id/time_to_tca rows are not allowed")
    final_rows = (
        frame.sort_values(["event_id", "time_to_tca"])
        .drop_duplicates("event_id", keep="first")
        .loc[:, ["event_id", "risk"]]
    )
    labels = final_rows.loc[:, ["event_id"]].assign(
        y=(final_rows["risk"] >= FINAL_RISK_THRESHOLD).astype("int8")
    )
    return labels.sort_values("event_id").reset_index(drop=True)


def split_event_ids(labels: pd.DataFrame) -> dict[str, set]:
    if labels["event_id"].duplicated().any() or not labels["y"].isin([0, 1]).all():
        raise ValueError("labels must contain one binary outcome per event_id")
    development, held_out = train_test_split(
        labels["event_id"].to_numpy(),
        test_size=0.4,
        random_state=SEED_STAGE1,
        stratify=labels["y"].to_numpy(),
    )
    held_out_labels = labels.set_index("event_id").loc[held_out, "y"].to_numpy()
    calibration, evaluation = train_test_split(
        held_out,
        test_size=0.5,
        random_state=SEED_STAGE2,
        stratify=held_out_labels,
    )
    return {
        "development": set(development),
        "calibration": set(calibration),
        "evaluation": set(evaluation),
    }


def build_partitions(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    labels = event_labels(frame)
    if len(labels) != EXPECTED_EVENTS or int(labels["y"].sum()) != EXPECTED_POSITIVES:
        raise ValueError("Unexpected event-label distribution")
    split_ids = split_event_ids(labels)
    labelled = frame.merge(labels, on="event_id", how="left", validate="many_to_one")
    decision_rows = labelled.loc[
        labelled["time_to_tca"].between(2.0, 7.0, inclusive="both")
    ].copy()
    development = decision_rows.loc[
        decision_rows["event_id"].isin(split_ids["development"])
    ].copy()
    calibration = decision_rows.loc[
        decision_rows["event_id"].isin(split_ids["calibration"])
    ].copy()
    evaluation = decision_rows.loc[
        decision_rows["event_id"].isin(split_ids["evaluation"])
    ].copy()
    return {
        "training": development,
        "calibration": calibration,
        "calibration_labels": labels.loc[
            labels["event_id"].isin(split_ids["calibration"])
        ].copy(),
        "evaluation_features": evaluation.drop(columns="y"),
        "evaluation_labels": labels.loc[
            labels["event_id"].isin(split_ids["evaluation"])
        ].copy(),
    }
