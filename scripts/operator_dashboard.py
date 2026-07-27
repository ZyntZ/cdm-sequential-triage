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
from confirmation import file_sha256, read_json, validate_policy

DECISIONS = ("ESCALATE", "MONITOR", "SAFE-EXCLUDE")
DECISION_CLASS = {"ESCALATE": "escalate", "MONITOR": "monitor", "SAFE-EXCLUDE": "safe"}
REQUIRED = {
    "event_id", "sequence_number", "time_to_tca", "score", "decision", "reason",
    "shift_score", "shift_gate_allowed", "eligible_history_count", "scores_sha256",
    "calibration_sha256", "model_sha256", "runtime_configuration_sha256",
    "safe_threshold", "escalation_threshold", "minimum_history",
    "min_days_to_tca", "max_days_to_tca", "is_current_decision",
}
HASH_COLUMNS = (
    "scores_sha256", "calibration_sha256", "model_sha256", "shift_gate_sha256",
    "model_manifest_sha256", "runtime_checkpoint_sha256",
    "runtime_configuration_sha256",
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _event_key(value: Any) -> str:
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
    runtime_hashes = combined["runtime_configuration_sha256"].astype(str).unique()
    if len(runtime_hashes) != 1:
        raise ValueError("Audit files use different runtime configurations")
    expected_runtime = {
        "safe_threshold": float(calibration["calibration"]["threshold"]),
        "minimum_history": int(calibration["policy"]["minimum_history"]),
        "min_days_to_tca": float(calibration["policy"]["min_days_to_tca"]),
        "max_days_to_tca": float(calibration["policy"]["max_days_to_tca"]),
    }
    for column, expected in expected_runtime.items():
        values = pd.to_numeric(combined[column], errors="coerce")
        if values.isna().any() or not np.allclose(values.to_numpy(), expected, rtol=0.0, atol=0.0):
            raise ValueError(f"Audit runtime {column} does not match the calibration artifact")
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


def build_dashboard(audit_paths: list[Path], calibration_path: Path, output_path: Path, confirmation_path: Path | None = None, max_events: int = 250) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"Dashboard output already exists: {output_path}")
    if max_events < 1:
        raise ValueError("max_events must be positive")
    calibration = read_json(calibration_path)
    policy = validate_policy(calibration.get("policy"))
    rule = calibration.get("calibration")
    if not isinstance(rule, dict) or "threshold" not in rule:
        raise ValueError("Calibration artifact has no threshold")
    audit = load_audits(audit_paths, calibration, calibration_path)
    current = current_events(audit)
    shown = current.head(max_events)
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
    selected = set(shown["__event_key"])
    for key, frame in audit[audit["__event_key"].isin(selected)].groupby("__event_key", sort=False):
        frame = frame.sort_values("sequence_number")
        timelines.append({"key": key, "label": str(frame.iloc[-1]["event_id"]), "updates": [
            {"sequence": int(r.sequence_number), "tca": float(r.time_to_tca), "score": float(r.score),
             "decision": str(r.decision), "reason": str(r.reason), "history": int(r.eligible_history_count),
             "gate": bool(r.shift_gate_allowed)} for r in frame.itertuples(index=False)
        ]})
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
        confirmation_html = f"""
<section class='panel evidence'><div class='panel-title'>LOCKED CONFIRMATION EVIDENCE</div><div class='evidence-grid'>
<div><span>Dangerous exclusions</span><strong>{int(metrics['danger_k'])}/{int(metrics['danger_n'])}</strong></div>
<div><span>Observed event rate</span><strong>{100*float(metrics['danger_rate']):.2f}%</strong></div>
<div class='{'pass' if passed else 'fail'}'><span>One-sided 95% UCB</span><strong>{100*float(metrics['danger_ucb']):.2f}%</strong><small>criterion ≤ {100*criterion:.2f}% · {'MET' if passed else 'NOT MET'}</small></div>
<div><span>Correct SAFE-EXCLUDE</span><strong>{100*float(metrics['safe_negative_rate']):.2f}%</strong></div>
<div><span>Median lead time</span><strong>{float(metrics['median_first_safe_tca']):.2f} d</strong></div></div>
<p class='caveat'>Historical confirmation_v1 evidence. The primary criterion was not met. This does not validate the preregistered v13 candidate.</p>
<div class='source'>confirmation {_escape(confirmation_path.name)} · sha256 {file_sha256(confirmation_path)}</div></section>"""

    policy_rows = "".join(f"<tr><th>{html.escape(k)}</th><td>{_escape(v)}</td></tr>" for k, v in policy.items())
    policy_rows += f"<tr><th>threshold</th><td>{_escape(rule.get('threshold'))}</td></tr><tr><th>calibration rank</th><td>{_escape(rule.get('rank'))} / {_escape(rule.get('n_positive'))}</td></tr><tr><th>PAC bound</th><td>{_escape(rule.get('pac_bound'))}</td></tr>"
    lineage = "".join(f"<tr><td>{_escape(p.name)}</td><td>{file_sha256(p)}</td></tr>" for p in audit_paths)
    lineage += f"<tr><td>{_escape(calibration_path.name)}</td><td>{file_sha256(calibration_path)}</td></tr>"
    runtime_configuration_sha256 = str(audit["runtime_configuration_sha256"].iloc[0])
    lineage += f"<tr><td>runtime configuration</td><td>{_escape(runtime_configuration_sha256)}</td></tr>"
    cards = "".join(f"<div class='kpi {DECISION_CLASS[d]}'><span>{d}</span><strong>{int(counts.get(d,0))}</strong><small>{100*counts.get(d,0)/events:.1f}% of current events</small></div>" for d in DECISIONS)

    document = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>CDM Triage Operator Console</title>
<style>:root{{--bg:#071018;--panel:#0c1822;--line:#20313d;--text:#e7f0f4;--muted:#8fa5b2;--safe:#51d18a;--monitor:#f1bb4b;--danger:#ff6677;--cyan:#48c7df}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at top right,#122b38,#071018 42%);color:var(--text);font:14px/1.45 Inter,Segoe UI,Arial,sans-serif}}main{{max-width:1500px;margin:auto;padding:24px}}header{{display:flex;justify-content:space-between;gap:24px;align-items:flex-end;border-bottom:1px solid var(--line);padding-bottom:18px}}h1{{margin:0;font-size:28px}}.eyebrow,.panel-title{{color:var(--cyan);font:12px monospace;letter-spacing:.15em}}.status{{border:1px solid var(--monitor);color:var(--monitor);padding:8px 12px;border-radius:4px;font-weight:700}}.grid{{display:grid;grid-template-columns:repeat(12,1fr);gap:16px;margin-top:16px}}.panel{{background:linear-gradient(180deg,#10222d,#09161f);border:1px solid var(--line);border-radius:8px;padding:18px;box-shadow:0 12px 30px #0003}}.summary,.evidence{{grid-column:span 12}}.active,.timeline{{grid-column:span 8}}.policy,.lineage{{grid-column:span 4}}.panel-title{{margin-bottom:14px;font-weight:700}}.kpis,.evidence-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}}.kpi,.evidence-grid div{{background:#09151e;padding:13px;border-left:3px solid var(--cyan)}}.kpi span,.kpi small,.evidence-grid span,.evidence-grid small{{display:block;color:var(--muted)}}.kpi strong,.evidence-grid strong{{font-size:25px}}.kpi.safe{{border-color:var(--safe)}}.kpi.monitor{{border-color:var(--monitor)}}.kpi.escalate,.evidence-grid .fail{{border-color:var(--danger)}}.evidence-grid .pass{{border-color:var(--safe)}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px 10px;border-bottom:1px solid #182a36;text-align:left}}th{{color:var(--muted);font:11px monospace}}.num{{font-family:monospace;text-align:right}}.event{{font-family:monospace}}.reason,.source{{color:var(--muted);font-size:12px}}.table-wrap{{overflow:auto;max-height:620px}}.badge{{padding:3px 7px;border-radius:3px;font:700 11px monospace}}.badge.safe{{color:var(--safe);background:#0f3024}}.badge.monitor{{color:var(--monitor);background:#332713}}.badge.escalate{{color:var(--danger);background:#341722}}select{{background:#09151e;color:var(--text);border:1px solid var(--line);padding:8px;width:100%;margin-bottom:12px}}.timeline-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px}}.step{{padding:10px;border:1px solid var(--line);background:#09151e}}.step strong,.step span{{display:block}}.step span{{color:var(--muted);font-size:12px}}.caveat{{border-left:3px solid var(--danger);padding:10px;background:#21131a}}.source{{margin-top:12px;font-family:monospace;word-break:break-all}}footer{{margin:22px 0;color:var(--muted);font-size:12px}}@media(max-width:980px){{.active,.policy,.timeline,.lineage{{grid-column:span 12}}.kpis,.evidence-grid{{grid-template-columns:1fr 1fr}}header{{flex-direction:column;align-items:flex-start}}}}</style></head><body><main>
<header><div><div class='eyebrow'>SPACE TRAFFIC · DECISION SUPPORT</div><h1>Sequential CDM Triage · Operator Console</h1><div>Calibrated event-level exclusion policy with auditable message-by-message decisions</div></div><div class='status'>HISTORICAL DEMO · NOT FOR OPERATIONS</div></header><div class='grid'>
<section class='panel summary'><div class='panel-title'>CURRENT RUNTIME STATE</div><div class='kpis'><div class='kpi'><span>Active events</span><strong>{events}</strong><small>across {len(audit_paths)} batch(es)</small></div><div class='kpi'><span>Processed updates</span><strong>{updates}</strong><small>message-level count</small></div>{cards}</div><div class='source'>gate: {'active · '+str(audit.loc[blocked,'__event_key'].nunique())+' events blocked' if gate_active else 'not active'} · showing {len(shown)} of {events} events</div></section>
<section class='panel active'><div class='panel-title'>ACTIVE EVENT QUEUE</div><div class='table-wrap'><table><thead><tr><th>Event</th><th>Current decision</th><th>Score</th><th>TCA, d</th><th>Seq</th><th>History</th><th>Gate</th><th>Reason</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>
<section class='panel policy'><div class='panel-title'>FROZEN POLICY</div><table>{policy_rows}</table><p class='caveat'>SAFE-EXCLUDE removes an event from the current manual-review queue while automated ingestion continues. It is not a maneuver command.</p></section>
<section class='panel timeline'><div class='panel-title'>EVENT DECISION TIMELINE</div><select id='event-select'></select><div id='timeline' class='timeline-grid'></div></section>
<section class='panel lineage'><div class='panel-title'>ARTIFACT LINEAGE</div><table><thead><tr><th>Artifact</th><th>SHA-256</th></tr></thead><tbody>{lineage}</tbody></table><div class='source'>model {_escape(calibration.get('model_sha256'))}<br>shift gate {_escape(calibration.get('shift_gate_sha256'))}</div></section>{confirmation_html}</div>
<footer>Dataset: ESA Collision Avoidance Challenge, Zenodo 10.5281/zenodo.4463683, CC BY 4.0. Target is high final calculated collision probability, not collision occurrence. Statistical control requires event-level exchangeability and is not an operational guarantee under arbitrary distribution shift.</footer>
<script>const events={timeline_json};const select=document.getElementById('event-select'),timeline=document.getElementById('timeline');function esc(s){{return String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}for(const e of events){{const o=document.createElement('option');o.value=e.key;o.textContent=e.label;select.appendChild(o);}}function render(){{const e=events.find(x=>x.key===select.value)||events[0];timeline.innerHTML='';if(!e)return;for(const u of e.updates){{const d=document.createElement('div');d.className='step';d.innerHTML=`<strong>${{esc(u.decision)}}</strong><span>seq ${{u.sequence}} · TCA ${{u.tca.toFixed(3)}} d</span><span>score ${{u.score.toPrecision(5)}} · history ${{u.history}}</span><span>${{esc(u.reason)}} · gate ${{u.gate?'ALLOW':'BLOCK'}}</span>`;timeline.appendChild(d);}}}}select.addEventListener('change',render);render();</script></main></body></html>"""
    _atomic_write(output_path, document)
    return {"updates": updates, "events": events, "current_decisions": {d: int(counts.get(d,0)) for d in DECISIONS}, "gate_active": bool(gate_active), "shown_events": len(shown), "confirmation": confirmation_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a static CDM triage operator dashboard")
    parser.add_argument("--audit", type=Path, nargs="+", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-events", type=int, default=250)
    args = parser.parse_args()
    print(json.dumps(build_dashboard(args.audit, args.calibration, args.output, args.confirmation, args.max_events), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
