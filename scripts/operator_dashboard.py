"""Build a self-contained operator dashboard from replay audit parquet files."""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from confirmation import (
    file_sha256, read_json, validate_calibration_artifact,
)
from triage import Decision, SequentialTriagePolicy

DECISIONS = ("ESCALATE", "MONITOR", "SAFE-EXCLUDE")
DECISION_CLASS = {"ESCALATE": "escalate", "MONITOR": "monitor", "SAFE-EXCLUDE": "safe"}
REQUIRED = {
    "event_id", "sequence_number", "time_to_tca", "score", "decision", "reason",
    "shift_score", "shift_gate_allowed", "decision_window_eligible",
    "eligible_history_count", "scores_sha256",
    "calibration_sha256", "model_sha256", "is_current_decision",
}
HASH_COLUMNS = (
    "scores_sha256", "calibration_sha256", "model_sha256", "shift_gate_sha256",
    "model_manifest_sha256", "runtime_checkpoint_sha256",
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _event_key(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    return f"{type(value).__name__}:{value}"


def _escape(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        value = f"{float(value):.5g}"
    return html.escape(str(value), quote=True)


def load_audits(paths: list[Path], calibration: dict, calibration_path: Path) -> pd.DataFrame:
    if not paths:
        raise ValueError("At least one replay audit is required")
    expected_calibration = file_sha256(calibration_path)
    expected_model = str(calibration.get("model_sha256"))
    frames = []
    for batch, path in enumerate(paths, start=1):
        frame = pd.read_parquet(path)
        missing = REQUIRED.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing audit columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{path} contains no audit rows")
        if not frame["decision"].isin(DECISIONS).all():
            raise ValueError(f"{path} contains an unsupported decision")
        if frame.duplicated(["event_id", "sequence_number"]).any():
            raise ValueError(f"{path} contains duplicate event sequence numbers")
        if not frame["calibration_sha256"].eq(expected_calibration).all():
            raise ValueError(f"{path} does not match the calibration artifact")
        if not frame["model_sha256"].astype(str).eq(expected_model).all():
            raise ValueError(f"{path} does not match the calibrated model")
        for column in HASH_COLUMNS:
            if column not in frame.columns:
                continue
            values = frame[column].dropna().astype(str).unique()
            if len(values) > 1:
                raise ValueError(f"{path} contains multiple {column} values")
            if len(values) == 1 and not HEX64.fullmatch(values[0]):
                raise ValueError(f"{path} contains an invalid {column}")
        frame = frame.copy()
        frame["audit_batch"] = batch
        frame["audit_file"] = path.name
        frame["audit_file_sha256"] = file_sha256(path)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["__event_key"] = combined["event_id"].map(_event_key)
    if combined.duplicated(["__event_key", "sequence_number"]).any():
        raise ValueError("Audit files overlap in event sequence numbers")
    return combined


def current_events(audit: pd.DataFrame) -> pd.DataFrame:
    ordered = audit.sort_values(["__event_key", "sequence_number", "audit_batch"], kind="mergesort")
    current = ordered.drop_duplicates("__event_key", keep="last").copy()
    current["__priority"] = current["decision"].map({"ESCALATE": 0, "MONITOR": 1, "SAFE-EXCLUDE": 2})
    return current.sort_values(["__priority", "time_to_tca", "score"], ascending=[True, True, False]).drop(columns="__priority")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)



SUPPORTED_LOCALES = {"en", "ru"}


def localize_operator_document(document: str, locale: str) -> str:
    if locale not in SUPPORTED_LOCALES:
        raise ValueError(f"Unsupported locale: {locale}")
    if locale == "en":
        return document
    replacements = [
        ("<html lang='en'>", "<html lang='ru'>"),
        ("CDM Triage Operator Console", "Операторская консоль триажа CDM"),
        ("Decision support for conjunction events", "Поддержка решений по событиям сближения"),
        ("Sequential CDM Triage · Operator Console", "Последовательный триаж CDM · Операторская консоль"),
        ("Calibrated event-level exclusion policy with auditable message-by-message decisions", "Калиброванная event-level политика с аудитом решения после каждого сообщения"),
        ("Historical replay · not for operations", "Историческое воспроизведение · не для эксплуатации"),
        ("Current state", "Текущее состояние"),
        ("Key figures", "Основные показатели"),
        ("This historical replay processed", "В историческом воспроизведении обработано"),
        ("CDM updates across", "обновлений CDM по"),
        ("event trajectories in the decision window.", "траекториям событий в рабочем окне."),
        ("Current queue:", "Текущая очередь:"),
        ("The locked confirmation observed", "В замороженном подтверждающем эксперименте получено"),
        ("dangerous exclusions; the one-sided 95% upper bound was", "опасных исключений; односторонняя 95%-я верхняя граница составила"),
        ("against the", "при"),
        ("criterion.", "критерии."),
        ("correct SAFE-EXCLUDE decisions per 1,000 evaluated events at a median lead of", "корректных решений SAFE-EXCLUDE на 1 000 оценённых событий при медианном упреждении"),
        ("The preregistered v13 candidate remains unopened and is not validated by confirmation_v1.", "Предзарегистрированный кандидат v13 остаётся нераскрытым и не подтверждается результатом confirmation_v1."),
        ("Example trajectory", "Пример траектории"),
        ("Selected by a frozen display rule: latest decision SAFE-EXCLUDE, a prior MONITOR step, then maximum lead time and trajectory length.", "Выбран фиксированным правилом отображения: последнее решение SAFE-EXCLUDE, ранее был MONITOR, затем максимальное упреждение и длина траектории."),
        ("First SAFE-EXCLUDE at message", "Первый SAFE-EXCLUDE на сообщении"),
        ("updates in the trajectory", "обновлений в траектории"),
        ("Shift-gate case", "Пример shift gate"),
        ("not available in this replay because confirmation_v1 has no fitted gate. No synthetic case is shown.", "недоступен в этом воспроизведении, поскольку в confirmation_v1 нет обученного gate. Синтетический пример не показывается."),
        ("Active events", "Активные события"), ("Processed updates", "Обработанные сообщения"),
        ("of current events", "текущих событий"), ("across", "в"), ("batch(es)", "пакетах"),
        ("message-level count", "число сообщений"), ("Batch chain", "Цепочка пакетов"),
        ("length", "длина"), ("showing", "показано"), ("events", "событий"),
        ("gate: active", "gate: активен"), ("events blocked", "событий заблокировано"),
        ("gate: not active", "gate: не активен"),
        ("Event queue", "Очередь событий"),
        ("Current decision", "Текущее решение"), ("Event", "Событие"),
        ("Score", "Score"), ("History", "История"), ("Reason", "Причина"),
        ("Policy settings", "Параметры политики"),
        ("SAFE-EXCLUDE removes an event from the current manual-review queue while automated ingestion continues. It is not a maneuver command.", "SAFE-EXCLUDE исключает событие из текущей очереди ручного анализа, но автоматический приём новых CDM продолжается. Это не команда на манёвр."),
        ("Decision history", "История решений"),
        ("Decision explanation", "Объяснение решения"),
        ("First SAFE-EXCLUDE", "Первый SAFE-EXCLUDE"),
        ("Artifact checksums", "Контрольные суммы артефактов"),
        ("Artifact", "Артефакт"), ("shift gate", "shift gate"),
        ("Processed batches", "Обработанные пакеты"),
        ("CDM rows", "Строки CDM"), ("Min TCA, d", "Мин. TCA, сут."),
        ("Max TCA, d", "Макс. TCA, сут."), ("Audit rows", "Строки аудита"),
        ("Previous entry", "Предыдущая запись"),
        ("Confirmation result", "Результат подтверждающего эксперимента"),
        ("Dangerous exclusions", "Опасные исключения"),
        ("Observed event rate", "Наблюдаемая доля"),
        ("One-sided 95% UCB", "Односторонняя 95%-я UCB"),
        ("criterion ≤", "критерий ≤"), ("NOT MET", "НЕ ВЫПОЛНЕН"), ("MET", "ВЫПОЛНЕН"),
        ("Correct SAFE-EXCLUDE", "Корректные SAFE-EXCLUDE"),
        ("Median lead time", "Медианное упреждение"),
        ("Historical confirmation_v1 evidence. The primary criterion was not met. This does not validate the preregistered v13 candidate.", "Историческое доказательство confirmation_v1. Основной критерий не выполнен. Результат не подтверждает предзарегистрированный кандидат v13."),
        ("Dataset:", "Набор данных:"),
        ("Target is high final calculated collision probability, not collision occurrence. Statistical control requires event-level exchangeability and is not an operational guarantee under arbitrary distribution shift.", "Целью является высокая финальная расчётная вероятность столкновения, а не факт столкновения. Статистический контроль требует event-level обменности и не является эксплуатационной гарантией при произвольном сдвиге распределения."),
    ]
    for source, target in replacements:
        document = document.replace(source, target)
    return document


REASON_EXPLANATIONS = {
    "score_at_or_above_escalation_threshold": (
        "Score reached the operational escalation threshold; manual review is required."
    ),
    "decision_window_not_open": (
        "The event is earlier than the calibrated decision window; automated monitoring continues."
    ),
    "decision_window_closed": (
        "The calibrated decision window has closed; SAFE-EXCLUDE is no longer permitted."
    ),
    "minimum_history_not_reached": (
        "Too few CDM updates are available inside the decision window; monitoring continues."
    ),
    "safe_exclude_blocked_by_shift_gate": (
        "The score crossed the safe threshold, but the applicability gate blocked SAFE-EXCLUDE."
    ),
    "score_at_or_below_calibrated_threshold": (
        "The calibrated safe threshold, history rule, decision window, and applicability checks passed."
    ),
    "score_between_decision_thresholds": (
        "The score did not cross a decision threshold; the event remains under monitoring."
    ),
}


def explain_event_sequence(
    event_id: Any,
    audit: pd.DataFrame,
    policy: dict[str, Any],
    threshold: float,
    escalation_threshold: float | None = None,
) -> dict[str, Any]:
    """Build a deterministic explanation from the accepted runtime audit only."""
    required = {
        "event_id", "sequence_number", "time_to_tca", "score", "decision",
        "reason", "shift_score", "shift_gate_allowed",
        "decision_window_eligible", "eligible_history_count",
    }
    missing = required.difference(audit.columns)
    if missing:
        raise ValueError(f"Audit is missing explanation columns: {sorted(missing)}")
    event_key = _event_key(event_id)
    keys = audit["event_id"].map(_event_key)
    event = audit.loc[keys.eq(event_key)].sort_values(
        "sequence_number", kind="mergesort"
    )
    if event.empty:
        raise ValueError(f"Event is not present in the audit: {event_id!r}")
    if event["sequence_number"].duplicated().any():
        raise ValueError(f"Event has duplicate sequence numbers: {event_id!r}")

    minimum_history = int(policy["minimum_history"])
    min_days = float(policy["min_days_to_tca"])
    max_days = float(policy["max_days_to_tca"])
    safe_threshold = float(threshold)
    steps = []
    for row in event.itertuples(index=False):
        reason = str(row.reason)
        if reason not in REASON_EXPLANATIONS:
            raise ValueError(f"Unsupported runtime decision reason: {reason}")
        shift_score = None
        if row.shift_score is not None and not pd.isna(row.shift_score):
            shift_score = float(row.shift_score)
        steps.append({
            "sequence_number": int(row.sequence_number),
            "time_to_tca": float(row.time_to_tca),
            "score": float(row.score),
            "decision": str(row.decision),
            "reason": reason,
            "eligible_history_count": int(row.eligible_history_count),
            "shift_gate_allowed": bool(row.shift_gate_allowed),
            "shift_score": shift_score,
            "decision_window_eligible": bool(row.decision_window_eligible),
            "history_sufficient": int(row.eligible_history_count) >= minimum_history,
            "score_at_or_below_safe_threshold": float(row.score) <= safe_threshold,
            "score_at_or_above_escalation_threshold": (
                False if escalation_threshold is None
                else float(row.score) >= float(escalation_threshold)
            ),
            "explanation": REASON_EXPLANATIONS[reason],
        })
    safe_steps = [
        step for step in steps if step["decision"] == Decision.SAFE_EXCLUDE.value
    ]
    return {
        "event_id": event_id,
        "total_updates": len(steps),
        "current_decision": steps[-1]["decision"],
        "first_safe_exclude_sequence": (
            None if not safe_steps else safe_steps[0]["sequence_number"]
        ),
        "first_safe_exclude_tca": (
            None if not safe_steps else safe_steps[0]["time_to_tca"]
        ),
        "policy": {
            "safe_threshold": safe_threshold,
            "escalation_threshold": escalation_threshold,
            "minimum_history": minimum_history,
            "min_days_to_tca": min_days,
            "max_days_to_tca": max_days,
        },
        "steps": steps,
    }

def select_showcase_monitor_to_safe(audit: pd.DataFrame) -> dict[str, Any]:
    """Select a reproducible MONITOR-to-SAFE-EXCLUDE trajectory for display."""
    ordered = audit.sort_values(
        ["__event_key", "sequence_number", "audit_batch"], kind="mergesort"
    )
    current = ordered.drop_duplicates("__event_key", keep="last")
    candidates: list[dict[str, Any]] = []
    for key in current.loc[
        current["decision"].eq(Decision.SAFE_EXCLUDE.value), "__event_key"
    ]:
        rows = ordered.loc[ordered["__event_key"].eq(key)]
        safe = rows.loc[rows["decision"].eq(Decision.SAFE_EXCLUDE.value)]
        if safe.empty:
            continue
        first = safe.iloc[0]
        prior = rows.loc[rows["sequence_number"] < first["sequence_number"]]
        if not prior["decision"].eq(Decision.MONITOR.value).any():
            continue
        candidates.append({
            "event_id": (
                rows.iloc[-1]["event_id"].item()
                if isinstance(rows.iloc[-1]["event_id"], np.generic)
                else rows.iloc[-1]["event_id"]
            ),
            "event_key": str(key),
            "first_safe_sequence": int(first["sequence_number"]),
            "first_safe_tca": float(first["time_to_tca"]),
            "total_updates": int(len(rows)),
        })
    if not candidates:
        raise ValueError("No MONITOR-to-SAFE-EXCLUDE trajectory exists in the audit")
    candidates.sort(key=lambda row: (
        -row["first_safe_tca"], -row["total_updates"], str(row["event_id"])
    ))
    return candidates[0]


def select_showcase_gate_blocked(audit: pd.DataFrame) -> dict[str, Any] | None:
    """Select a reproducible shift-gate block, or return None if none occurred."""
    blocked = audit["reason"].eq("safe_exclude_blocked_by_shift_gate")
    if not blocked.any():
        return None
    ordered = audit.sort_values(
        ["__event_key", "sequence_number", "audit_batch"], kind="mergesort"
    )
    current = ordered.drop_duplicates("__event_key", keep="last").set_index("__event_key")
    candidates: list[dict[str, Any]] = []
    for key in ordered.loc[blocked, "__event_key"].unique():
        rows = ordered.loc[ordered["__event_key"].eq(key)]
        blocked_rows = rows.loc[rows["reason"].eq("safe_exclude_blocked_by_shift_gate")]
        shift_scores = blocked_rows["shift_score"].dropna()
        candidates.append({
            "event_id": (
                rows.iloc[-1]["event_id"].item()
                if isinstance(rows.iloc[-1]["event_id"], np.generic)
                else rows.iloc[-1]["event_id"]
            ),
            "event_key": str(key),
            "current_decision": str(current.loc[key, "decision"]),
            "blocked_updates": int(len(blocked_rows)),
            "max_shift_score": (
                None if shift_scores.empty else float(shift_scores.max())
            ),
        })
    candidates.sort(key=lambda row: (
        0 if row["current_decision"] == Decision.MONITOR.value else 1,
        -row["blocked_updates"],
        -(row["max_shift_score"] if row["max_shift_score"] is not None else float("-inf")),
        str(row["event_id"]),
    ))
    return candidates[0]


def build_dashboard(
    audit_paths: list[Path],
    calibration_path: Path,
    output_path: Path,
    confirmation_path: Path | None = None,
    max_events: int = 250,
    checkpoint_path: Path | None = None,
    max_chain_rows: int = 50,
    locale: str = "en",
) -> dict[str, Any]:
    if locale not in SUPPORTED_LOCALES:
        raise ValueError(f"Unsupported locale: {locale}")
    if output_path.exists():
        raise FileExistsError(f"Dashboard output already exists: {output_path}")
    if max_events < 1:
        raise ValueError("max_events must be positive")
    if max_chain_rows < 1:
        raise ValueError("max_chain_rows must be positive")
    calibration = read_json(calibration_path)
    policy, rule = validate_calibration_artifact(calibration)
    audit = load_audits(audit_paths, calibration, calibration_path)
    chain_summary = None
    chain: list[dict[str, Any]] = []
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Runtime checkpoint does not exist: {checkpoint_path}")
        runtime = SequentialTriagePolicy.restore(checkpoint_path)
        envelope = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint_digest = envelope.get("checkpoint_sha256")
        if not isinstance(checkpoint_digest, str) or not HEX64.fullmatch(checkpoint_digest):
            raise ValueError("Runtime checkpoint has no valid payload digest")
        last_batch = audit.loc[audit["audit_batch"] == audit["audit_batch"].max()]
        audit_digests = (
            last_batch["runtime_checkpoint_sha256"].dropna().astype(str).unique()
        )
        if len(audit_digests) != 1 or audit_digests[0] != checkpoint_digest:
            raise ValueError("Runtime checkpoint does not match the latest replay audit")
        chain = runtime.processed_batches()
        chain_head = runtime.processed_batch_chain_head()
        if not chain or chain_head is None:
            raise ValueError("Runtime checkpoint has no processed batch chain")
        chain_summary = {
            "status": "VERIFIED",
            "length": len(chain),
            "head_sha256": chain_head,
            "checkpoint_sha256": checkpoint_digest,
            "checkpoint_file_sha256": file_sha256(checkpoint_path),
            "displayed_rows": min(len(chain), max_chain_rows),
            "clipped": len(chain) > max_chain_rows,
        }
    current = current_events(audit)
    try:
        monitor_to_safe_showcase = select_showcase_monitor_to_safe(audit)
    except ValueError:
        monitor_to_safe_showcase = None
    gate_blocked_showcase = select_showcase_gate_blocked(audit)
    shown = current.head(max_events)
    timeline_current = shown
    if (
        monitor_to_safe_showcase is not None
        and monitor_to_safe_showcase["event_key"] not in set(shown["__event_key"])
    ):
        showcase_current = current.loc[
            current["__event_key"].eq(monitor_to_safe_showcase["event_key"])
        ]
        timeline_current = pd.concat([shown, showcase_current], ignore_index=True)
    counts = current["decision"].value_counts()
    events, updates = len(current), len(audit)
    gate_active = audit["shift_score"].notna().any()
    blocked = ~audit["shift_gate_allowed"].astype(bool) if gate_active else pd.Series(False, index=audit.index)

    rows = []
    for row in shown.itertuples(index=False):
        decision = str(row.decision)
        rows.append(
            "<tr>" +
            f"<td class='event'>{_escape(row.event_id)}</td>" +
            f"<td><span class='badge {DECISION_CLASS[decision]}'>{_escape(decision)}</span></td>" +
            f"<td class='num'>{_escape(row.score)}</td><td class='num'>{_escape(row.time_to_tca)}</td>" +
            f"<td class='num'>{int(row.sequence_number)}</td><td class='num'>{int(row.eligible_history_count)}</td>" +
            f"<td>{'ALLOW' if bool(row.shift_gate_allowed) else 'BLOCK'}</td><td class='reason'>{_escape(row.reason)}</td></tr>"
        )

    timelines = []
    shown_keys = timeline_current["__event_key"].tolist()
    for key in shown_keys:
        frame = audit.loc[audit["__event_key"].eq(key)].sort_values(
            "sequence_number", kind="mergesort"
        )
        event_id = frame.iloc[-1]["event_id"]
        explanation = explain_event_sequence(
            event_id, frame, policy, float(rule["threshold"]),
            escalation_threshold=(
                None
                if "escalation_threshold" not in frame
                or frame["escalation_threshold"].dropna().empty
                else float(frame["escalation_threshold"].dropna().iloc[0])
            ),
        )
        timelines.append({
            "key": key,
            "label": str(event_id),
            "first_safe_sequence": explanation["first_safe_exclude_sequence"],
            "first_safe_tca": explanation["first_safe_exclude_tca"],
            "updates": [
                {
                    "sequence": step["sequence_number"],
                    "tca": step["time_to_tca"],
                    "score": step["score"],
                    "decision": step["decision"],
                    "reason": step["reason"],
                    "history": step["eligible_history_count"],
                    "gate": step["shift_gate_allowed"],
                    "explanation": step["explanation"],
                }
                for step in explanation["steps"]
            ],
        })
    timeline_json = json.dumps(timelines, ensure_ascii=False).replace("</", "<\\/")

    confirmation_html = ""
    confirmation_summary = None
    if confirmation_path is not None:
        confirmation = read_json(confirmation_path)
        metrics = confirmation.get("evaluation", {})
        required_metrics = {"danger_k", "danger_n", "danger_rate", "danger_ucb", "safe_negative_rate", "median_first_safe_tca"}
        missing = required_metrics.difference(metrics)
        if missing:
            raise ValueError(f"Confirmation artifact is missing metrics: {sorted(missing)}")
        criterion = float(policy["alpha"])
        passed = float(metrics["danger_ucb"]) <= criterion
        confirmation_summary = {"danger_k": int(metrics["danger_k"]), "danger_n": int(metrics["danger_n"]), "danger_ucb": float(metrics["danger_ucb"]), "criterion": criterion, "passed": passed}
        if "evaluation_events" in confirmation and "safe_negative" in metrics:
            evaluation_events = int(confirmation["evaluation_events"])
            if evaluation_events <= 0:
                raise ValueError("Confirmation evaluation_events must be positive")
            confirmation_summary["correct_safe_excludes_per_1000"] = (
                1000.0 * int(metrics["safe_negative"]) / evaluation_events
            )
            confirmation_summary["median_first_safe_tca"] = float(metrics["median_first_safe_tca"])
        confirmation_html = f"""
<section class='panel evidence'><div class='panel-title'>Confirmation result</div><div class='evidence-grid'>
<div><span>Dangerous exclusions</span><strong>{int(metrics['danger_k'])}/{int(metrics['danger_n'])}</strong></div>
<div><span>Observed event rate</span><strong>{100*float(metrics['danger_rate']):.2f}%</strong></div>
<div class='{'pass' if passed else 'fail'}'><span>One-sided 95% UCB</span><strong>{100*float(metrics['danger_ucb']):.2f}%</strong><small>criterion ≤ {100*criterion:.2f}% · {'MET' if passed else 'NOT MET'}</small></div>
<div><span>Correct SAFE-EXCLUDE</span><strong>{100*float(metrics['safe_negative_rate']):.2f}%</strong></div>
<div><span>Median lead time</span><strong>{float(metrics['median_first_safe_tca']):.2f} d</strong></div></div>
<p class='caveat'>Historical confirmation_v1 evidence. The primary criterion was not met. This does not validate the preregistered v13 candidate.</p>
<div class='source'>confirmation {_escape(confirmation_path.name)} · sha256 {file_sha256(confirmation_path)}</div></section>"""

    decision_breakdown = " · ".join(
        f"{decision} {int(counts.get(decision, 0)):,}" for decision in DECISIONS
    )
    briefing_confirmation = ""
    if confirmation_summary is not None:
        result = "MET" if confirmation_summary["passed"] else "NOT MET"
        briefing_confirmation = (
            "<p>The locked confirmation observed "
            f"<strong>{confirmation_summary['danger_k']}/{confirmation_summary['danger_n']}</strong> "
            "dangerous exclusions; the one-sided 95% upper bound was "
            f"<strong>{100*confirmation_summary['danger_ucb']:.2f}%</strong> against the "
            f"<strong>{100*confirmation_summary['criterion']:.2f}%</strong> criterion: "
            f"<strong>{result}</strong>.</p>"
        )
        if "correct_safe_excludes_per_1000" in confirmation_summary:
            briefing_confirmation += (
                f"<p><strong>{confirmation_summary['correct_safe_excludes_per_1000']:.0f}</strong> "
                "correct SAFE-EXCLUDE decisions per 1,000 evaluated events at a median lead of "
                f"<strong>{confirmation_summary['median_first_safe_tca']:.2f} d</strong>.</p>"
            )
        briefing_confirmation += (
            "<p class='caveat'>The preregistered v13 candidate remains unopened and "
            "is not validated by confirmation_v1.</p>"
        )
    briefing_html = (
        "<section id='judge-briefing' class='panel summary briefing'>"
        "<div class='panel-title'>Key figures</div>"
        f"<p>This historical replay processed <strong>{updates:,}</strong> CDM updates "
        f"across <strong>{events:,}</strong> event trajectories in the decision window.</p>"
        f"<p>Current queue: <strong>{decision_breakdown}</strong>.</p>"
        f"{briefing_confirmation}</section>"
    )
    gate_case_text = (
        "not available in this replay because confirmation_v1 has no fitted gate. "
        "No synthetic case is shown."
        if gate_blocked_showcase is None
        else f"event {_escape(gate_blocked_showcase['event_id'])} · "
             f"{gate_blocked_showcase['blocked_updates']} blocked updates"
    )
    showcase_html = ""
    if monitor_to_safe_showcase is not None:
        showcase_html = (
            "<section id='deterministic-showcase' class='panel summary briefing'>"
            "<div class='panel-title'>Example trajectory</div>"
            "<p>Selected by a frozen display rule: latest decision SAFE-EXCLUDE, a prior "
            "MONITOR step, then maximum lead time and trajectory length.</p>"
            f"<p>Event <strong>{_escape(monitor_to_safe_showcase['event_id'])}</strong> · "
            f"First SAFE-EXCLUDE at message <strong>{monitor_to_safe_showcase['first_safe_sequence']}</strong> · "
            f"TCA <strong>{monitor_to_safe_showcase['first_safe_tca']:.3f} d</strong> · "
            f"<strong>{monitor_to_safe_showcase['total_updates']}</strong> updates in the trajectory.</p>"
            f"<p class='caveat'>Shift-gate case: {gate_case_text}</p></section>"
        )

    policy_rows = "".join(f"<tr><th>{html.escape(k)}</th><td>{_escape(v)}</td></tr>" for k, v in policy.items())
    policy_rows += f"<tr><th>threshold</th><td>{_escape(rule.get('threshold'))}</td></tr><tr><th>calibration rank</th><td>{_escape(rule.get('rank'))} / {_escape(rule.get('n_positive'))}</td></tr><tr><th>PAC bound</th><td>{_escape(rule.get('pac_bound'))}</td></tr>"
    lineage = "".join(f"<tr><td>{_escape(p.name)}</td><td>{file_sha256(p)}</td></tr>" for p in audit_paths)
    lineage += f"<tr><td>{_escape(calibration_path.name)}</td><td>{file_sha256(calibration_path)}</td></tr>"
    if chain_summary is not None:
        lineage += (
            f"<tr><td>{_escape(checkpoint_path.name)}</td>"
            f"<td>{_escape(chain_summary['checkpoint_file_sha256'])}</td></tr>"
        )
    cards = "".join(f"<div class='kpi {DECISION_CLASS[d]}'><span>{d}</span><strong>{int(counts.get(d,0))}</strong><small>{100*counts.get(d,0)/events:.1f}% of current events</small></div>" for d in DECISIONS)
    chain_card = ""
    if chain_summary is not None:
        chain_card = (
            "<div class='kpi safe'><span>Batch chain</span>"
            f"<strong>{chain_summary['status']}</strong>"
            f"<small>length {chain_summary['length']} · head "
            f"{_escape(chain_summary['head_sha256'][:16])}…</small></div>"
        )

    chain_table_html = ""
    if chain_summary is not None:
        displayed_chain = chain[-max_chain_rows:]
        chain_rows = []
        for index, entry in enumerate(
            displayed_chain, start=len(chain) - len(displayed_chain) + 1
        ):
            previous_hash = entry.get("previous_entry_sha256")
            previous_label = (
                "GENESIS" if previous_hash is None
                else f"{_escape(str(previous_hash)[:16])}…"
            )
            chain_rows.append(
                "<tr>"
                f"<td class='num'>{index}</td>"
                f"<td class='num'>{int(entry['rows'])}</td>"
                f"<td class='num'>{int(entry['events'])}</td>"
                f"<td class='num'>{float(entry['min_time_to_tca']):.3f}</td>"
                f"<td class='num'>{float(entry['max_time_to_tca']):.3f}</td>"
                f"<td class='num'>{int(entry['first_audit_row'])}–{int(entry['last_audit_row'])}</td>"
                f"<td class='event'>{_escape(str(entry['scores_sha256'])[:16])}…</td>"
                f"<td class='event'>{_escape(str(entry['entry_sha256'])[:16])}…</td>"
                f"<td class='event'>{previous_label}</td>"
                "</tr>"
            )
        clipped_note = (
            f" · showing last {len(displayed_chain)} of {len(chain)} batches"
            if chain_summary["clipped"] else ""
        )
        chain_table_html = f"""
<section class='panel summary'><div class='panel-title'>Processed batches</div>
<div class='table-wrap'><table><thead><tr><th>#</th><th>CDM rows</th><th>Events</th><th>Min TCA, d</th><th>Max TCA, d</th><th>Audit rows</th><th>Scores SHA-256</th><th>Entry SHA-256</th><th>Previous entry</th></tr></thead><tbody>{''.join(chain_rows)}</tbody></table></div>
<div class='source'>status {_escape(chain_summary['status'])} · head {_escape(chain_summary['head_sha256'])}{clipped_note}</div></section>"""

    document = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>CDM Triage Operator Console</title>
<style>:root{{--bg:#f4f5f7;--panel:#fff;--line:#d8dce2;--text:#20242a;--muted:#68707c;--safe:#246b45;--monitor:#8a5a00;--danger:#a12638;--accent:#315f8c}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}}main{{max-width:1380px;margin:auto;padding:24px}}header{{display:flex;justify-content:space-between;gap:24px;align-items:flex-end;border-bottom:1px solid var(--line);padding-bottom:16px}}h1{{margin:0;font-size:26px;font-weight:600}}.eyebrow{{color:var(--muted);font-size:12px;margin-bottom:4px}}.status{{border:1px solid #d8a4ac;color:var(--danger);background:#fff7f8;padding:7px 10px;border-radius:3px;font-weight:600}}.grid{{display:grid;grid-template-columns:minmax(0,2fr) minmax(280px,1fr);gap:14px;margin-top:14px}}.panel{{min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:16px}}.summary,.evidence,.briefing{{grid-column:1/-1}}.active,.timeline{{grid-column:1}}.policy,.lineage{{grid-column:2}}.panel-title{{margin-bottom:12px;color:#394150;font-size:14px;font-weight:600}}.briefing p{{margin:7px 0;max-width:100ch}}.kpis,.evidence-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}}.kpi,.evidence-grid div{{min-width:0;background:#f7f8fa;padding:11px;border:1px solid #e4e7eb;border-left:3px solid var(--accent)}}.kpi span,.kpi small,.evidence-grid span,.evidence-grid small{{display:block;color:var(--muted)}}.kpi strong,.evidence-grid strong{{display:block;font-size:22px;overflow-wrap:anywhere}}.kpi.safe{{border-left-color:var(--safe)}}.kpi.monitor{{border-left-color:var(--monitor)}}.kpi.escalate,.evidence-grid .fail{{border-left-color:var(--danger)}}.evidence-grid .pass{{border-left-color:var(--safe)}}table{{width:100%;border-collapse:collapse;table-layout:auto}}th,td{{padding:8px;border-bottom:1px solid #e6e8ec;text-align:left;vertical-align:top;overflow-wrap:anywhere}}th{{color:var(--muted);font-size:11px;font-weight:600}}.num{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;text-align:right;white-space:nowrap}}.event{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}}.reason,.source{{color:var(--muted);font-size:12px;overflow-wrap:anywhere}}.table-wrap{{max-width:100%;overflow:auto;max-height:620px}}.badge{{display:inline-block;padding:3px 6px;border-radius:3px;font:600 11px ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap}}.badge.safe{{color:#185c38;background:#e8f3ed}}.badge.monitor{{color:#754c00;background:#fff4d6}}.badge.escalate{{color:#8f1e2f;background:#fbeaec}}select{{background:#fff;color:var(--text);border:1px solid #c9ced6;border-radius:3px;padding:8px;width:100%;margin-bottom:12px}}.timeline-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px}}.step{{min-width:0;padding:10px;border:1px solid #dfe3e8;border-radius:3px;background:#fafbfc;overflow-wrap:anywhere}}.step strong,.step span{{display:block}}.step span{{color:var(--muted);font-size:12px}}.caveat{{border-left:3px solid var(--danger);padding:9px 11px;background:#fff7f8;color:#4b2a30;overflow-wrap:anywhere}}.source{{margin-top:10px;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;word-break:break-word}}footer{{margin:20px 0 0;padding-top:12px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}}@media(max-width:900px){{main{{padding:14px}}header{{flex-direction:column;align-items:flex-start}}.grid{{grid-template-columns:1fr}}.active,.policy,.timeline,.lineage{{grid-column:1}}.kpis,.evidence-grid{{grid-template-columns:repeat(auto-fit,minmax(140px,1fr))}}}}@media print{{body{{background:#fff;color:#000}}main{{max-width:none;padding:0}}header{{border-color:#777}}.grid{{display:block}}.panel{{background:#fff;border:1px solid #aaa;box-shadow:none;break-inside:avoid;margin:0 0 10px}}.kpi,.evidence-grid div,.step{{background:#fff}}.status,.caveat,.source,.reason{{color:#000}}.caveat{{background:#fff;border-color:#8f1e2f}}select,footer,.active,.lineage{{display:none}}.timeline{{grid-column:span 12}}.table-wrap{{max-height:none;overflow:visible}}th,td{{border-color:#bbb}}.badge.safe{{color:#176b38;background:#e6f4ec}}.badge.monitor{{color:#6b5000;background:#fdf6e3}}.badge.escalate{{color:#8d001c;background:#fdecea}}}}</style></head><body><main>
<header><div><div class='eyebrow'>Decision support for conjunction events</div><h1>Sequential CDM Triage · Operator Console</h1><div>Calibrated event-level exclusion policy with auditable message-by-message decisions</div></div><div class='status'>Historical replay · not for operations</div></header><div class='grid'>
<section class='panel summary'><div class='panel-title'>Current state</div><div class='kpis'><div class='kpi'><span>Active events</span><strong>{events}</strong><small>across {len(audit_paths)} batch(es)</small></div><div class='kpi'><span>Processed updates</span><strong>{updates}</strong><small>message-level count</small></div>{cards}{chain_card}</div><div class='source'>gate: {'active · '+str(audit.loc[blocked,'__event_key'].nunique())+' events blocked' if gate_active else 'not active'} · showing {len(shown)} of {events} events</div></section>{briefing_html}{showcase_html}
<section class='panel active'><div class='panel-title'>Event queue</div><div class='table-wrap'><table><thead><tr><th>Event</th><th>Current decision</th><th>Score</th><th>TCA, d</th><th>Seq</th><th>History</th><th>Gate</th><th>Reason</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section class='panel policy'><div class='panel-title'>Policy settings</div><table>{policy_rows}</table><p class='caveat'>SAFE-EXCLUDE removes an event from the current manual-review queue while automated ingestion continues. It is not a maneuver command.</p></section>
<section class='panel timeline'><div class='panel-title'>Decision history</div><select id='event-select'></select><div id='event-summary' class='source'></div><div id='timeline' class='timeline-grid'></div></section>
<section class='panel lineage'><div class='panel-title'>Artifact checksums</div><table><thead><tr><th>Artifact</th><th>SHA-256</th></tr></thead><tbody>{lineage}</tbody></table><div class='source'>model {_escape(calibration.get('model_sha256'))}<br>shift gate {_escape(calibration.get('shift_gate_sha256'))}</div></section>{chain_table_html}{confirmation_html}</div>
<footer>Dataset: ESA Collision Avoidance Challenge, Zenodo 10.5281/zenodo.4463683, CC BY 4.0. Target is high final calculated collision probability, not collision occurrence. Statistical control requires event-level exchangeability and is not an operational guarantee under arbitrary distribution shift.</footer>
<script>const events={timeline_json};const select=document.getElementById('event-select'),timeline=document.getElementById('timeline'),eventSummary=document.getElementById('event-summary');function esc(s){{return String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}for(const e of events){{const o=document.createElement('option');o.value=e.key;o.textContent=e.label;select.appendChild(o);}}const showcaseKey={json.dumps(None if monitor_to_safe_showcase is None else monitor_to_safe_showcase['event_key'])};if([...select.options].some(o=>o.value===showcaseKey))select.value=showcaseKey;function render(){{const e=events.find(x=>x.key===select.value)||events[0];timeline.innerHTML='';eventSummary.textContent='';if(!e)return;eventSummary.textContent=e.first_safe_sequence===null?'First SAFE-EXCLUDE: none':`First SAFE-EXCLUDE: seq ${{e.first_safe_sequence}} · TCA ${{e.first_safe_tca.toFixed(3)}} d`;for(const u of e.updates){{const d=document.createElement('div');d.className='step';d.innerHTML=`<strong>${{esc(u.decision)}}</strong><span>seq ${{u.sequence}} · TCA ${{u.tca.toFixed(3)}} d</span><span>score ${{u.score.toPrecision(5)}} · history ${{u.history}}</span><span>${{esc(u.reason)}} · gate ${{u.gate?'ALLOW':'BLOCK'}}</span><span><b>Decision explanation:</b> ${{esc(u.explanation)}}</span>`;timeline.appendChild(d);}}}}select.addEventListener('change',render);render();</script></main></body></html>"""
    document = localize_operator_document(document, locale)
    _atomic_write(output_path, document)
    return {"updates": updates, "events": events, "current_decisions": {d: int(counts.get(d,0)) for d in DECISIONS}, "gate_active": bool(gate_active), "shown_events": len(shown), "chain": chain_summary, "confirmation": confirmation_summary, "showcase_monitor_to_safe": monitor_to_safe_showcase, "showcase_gate_blocked": gate_blocked_showcase, "locale": locale}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a static CDM triage operator dashboard")
    parser.add_argument("--audit", type=Path, nargs="+", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-events", type=int, default=250)
    parser.add_argument("--max-chain-rows", type=int, default=50)
    parser.add_argument("--locale", choices=sorted(SUPPORTED_LOCALES), default="en")
    args = parser.parse_args()
    print(json.dumps(build_dashboard(
        args.audit, args.calibration, args.output,
        confirmation_path=args.confirmation,
        max_events=args.max_events,
        checkpoint_path=args.checkpoint,
        max_chain_rows=args.max_chain_rows,
        locale=args.locale,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
