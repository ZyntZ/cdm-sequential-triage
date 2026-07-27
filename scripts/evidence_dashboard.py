"""Build a self-contained dashboard separating development, confirmation, and preregistered evidence."""
from __future__ import annotations
import argparse, html, json, os, tempfile
from pathlib import Path
from typing import Any
import pandas as pd
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from confirmation import file_sha256, read_json

def esc(x): return html.escape('—' if x is None else str(x),quote=True)
def pct(x): return f"{100*float(x):.2f}%"
def atomic_write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True); name=None
    try:
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,prefix=f'.{path.name}.',suffix='.tmp',delete=False) as f:
            name=f.name; f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(name,path); name=None
    finally:
        if name: Path(name).unlink(missing_ok=True)
def verify(path,expected,label):
    actual=file_sha256(path)
    if actual!=expected: raise ValueError(f"{label} SHA-256 mismatch")
    return actual

def build_evidence_dashboard(root:Path,output:Path)->dict[str,Any]:
    if output.exists(): raise FileExistsError(f"Evidence dashboard exists: {output}")
    a=root/'artifacts'; r=root/'reports'; c=a/'confirmation_v1'
    v10=read_json(a/'development_score_ensemble_v10.json'); v11=read_json(a/'development_score_ensemble_repeated_v11.json')
    conf=read_json(c/'confirmation.json'); cal=read_json(c/'calibration.json'); clock=read_json(c/'confirmation.lock')
    pre=read_json(a/'next_validation_preregistration_v12.json'); plock=read_json(a/'next_validation_preregistration_v12.lock')
    v13=read_json(a/'catboost_tail_aligned_final_v13.json'); plan=pd.read_csv(r/'next_validation_sample_size_v12.csv')
    if v10.get('evaluation_accessed') is not False or v11.get('evaluation_accessed') is not False: raise ValueError('Development artifact accessed evaluation data')
    prehash=verify(a/'next_validation_preregistration_v12.json',plock['preregistration_sha256'],'preregistration')
    planhash=verify(r/'next_validation_sample_size_v12.csv',plock['planning_sha256'],'planning')
    verify(a/'next_validation_preregistration_v12.lock',v13['preregistration']['lock_sha256'],'preregistration lock')
    modelhash=verify(a/'catboost_tail_aligned_final_v13.cbm',v13['outputs']['model']['sha256'],'v13 model')
    terminal=pre['candidate_selection']['terminal_development_artifacts']
    verify(a/'development_score_ensemble_v10.json',next(v for k,v in terminal.items() if k.endswith('development_score_ensemble_v10.json')),'v10')
    verify(a/'development_score_ensemble_repeated_v11.json',next(v for k,v in terminal.items() if k.endswith('development_score_ensemble_repeated_v11.json')),'v11')
    verify(c/'calibration.json',clock['calibration_artifact_sha256'],'confirmation calibration')
    verify(c/'evaluation_scores.parquet',clock['evaluation_scores_sha256'],'confirmation scores')
    if clock['calibration_artifact_sha256']!=conf['calibration_artifact_sha256'] or clock['evaluation_scores_sha256']!=conf['evaluation_scores_sha256']: raise ValueError('Confirmation lock/result mismatch')
    if v13['threshold'] is not None or v13['calibration_accessed'] is not False or v13['evaluation_accessed'] is not False: raise ValueError('v13 is not frozen before calibration')
    criterion=float(conf['policy']['alpha']); passed=float(conf['evaluation']['danger_ucb'])<=criterion
    devrows=''.join(f"<tr><td>{esc(x['method'])}</td><td>{x['danger_k']}/{x['danger_n']}</td><td>{pct(x['danger_ucb'])}</td><td>{pct(x['safe_negative_rate'])}</td><td>{'YES' if x['safety_feasible'] else 'NO'}</td><td>{'YES' if x['pareto_frontier'] else 'NO'}</td></tr>" for x in v10['summary'])
    stab=''.join(f"<tr><td>{esc(x['method'])}</td><td>{int(x['repeats'])}</td><td>{pct(x['median_safe_negative_rate'])}</td><td>{pct(x['coverage_delta_positive_fraction'])}</td><td>{pct(x['danger_not_worse_fraction'])}</td></tr>" for x in v11['summary'])
    pc='pass_probability_if_true_rate_0.05'; prow=''.join(f"<tr><td>{int(x['positive_events'])}</td><td>{int(x['maximum_passing_failures'])}</td><td>{pct(x['upper_bound_at_maximum'])}</td><td>{pct(x[pc])}</td></tr>" for _,x in plan.iterrows())
    m=conf['evaluation']; cr=cal['calibration']; cand=pre['candidate']; ns=pre['new_study']
    css="body{margin:0;background:#071018;color:#e6eef2;font:14px Arial}main{max-width:1350px;margin:auto;padding:28px}h1{margin:5px 0 20px}.tier{border:1px solid #263b48;border-left:5px solid;padding:20px;margin:18px 0;background:#0d1b25}.dev{border-left-color:#efb84b}.conf{border-left-color:#ff6375}.next{border-left-color:#48c7df}.tag{font:700 12px monospace;letter-spacing:.12em}.dev .tag{color:#efb84b}.conf .tag{color:#ff6375}.next .tag{color:#48c7df}.fail{background:#35151d;border:1px solid #ff6375;padding:12px;font-weight:bold}table{width:100%;border-collapse:collapse;margin:12px 0}th,td{padding:8px;border-bottom:1px solid #213541;text-align:left}th{color:#91a8b5;font:11px monospace}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.kpi{background:#09151e;padding:12px}.kpi strong,.kpi span{display:block}.kpi strong{font-size:22px}.caveat{color:#afbec6;border-left:3px solid #788c97;padding-left:10px}.hash{font:11px monospace;color:#8fa5b2;word-break:break-all}@media(max-width:800px){.grid{grid-template-columns:1fr 1fr}}"
    doc=f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>CDM Triage Evidence</title><style>{css}</style></head><body><main><div class='tag'>SCIENTIFIC EVIDENCE RECORD · NOT FOR OPERATIONS</div><h1>Sequential CDM Triage · Evidence Dashboard</h1>
<section id='development' class='tier dev'><div class='tag'>TIER 1 · EXPLORATORY DEVELOPMENT EVIDENCE</div><h2>Safety–automation frontier</h2><table><tr><th>Method</th><th>Danger</th><th>UCB95</th><th>Safe-negative</th><th>Feasible</th><th>Pareto</th></tr>{devrows}</table><h3>250 correlated calibration-split repeats</h3><table><tr><th>Method</th><th>Repeats</th><th>Median coverage</th><th>Coverage gain &gt; 0</th><th>Danger not worse</th></tr>{stab}</table><p class='caveat'>Development data only: 7,146 events, 192 positives. Fixed out-of-fold scores were reused across correlated calibration splits; these are not independent retraining replications and are not confirmation evidence.</p><div class='hash'>v10 {file_sha256(a/'development_score_ensemble_v10.json')} · v11 {file_sha256(a/'development_score_ensemble_repeated_v11.json')}</div></section>
<section id='confirmation' class='tier conf'><div class='tag'>TIER 2 · LOCKED CONFIRMATION_V1 · {'CRITERION MET' if passed else 'CRITERION NOT MET'}</div><h2>Snapshot-model confirmation</h2><div class='fail'>PRE-SPECIFIED CRITERION {'MET' if passed else 'NOT MET'}: UCB {pct(m['danger_ucb'])} {'≤' if passed else '>'} α {pct(criterion)}.</div><div class='grid'><div class='kpi'><span>Dangerous exclusions</span><strong>{m['danger_k']}/{m['danger_n']}</strong></div><div class='kpi'><span>Observed danger</span><strong>{pct(m['danger_rate'])}</strong></div><div class='kpi'><span>Safe-negative</span><strong>{pct(m['safe_negative_rate'])}</strong></div><div class='kpi'><span>Median lead</span><strong>{m['median_first_safe_tca']:.2f} d</strong></div></div><p>PAC rank {cr['rank']}/{cr['n_positive']} · threshold {cr['threshold']:.8g} · calibration bound {pct(cr['pac_bound'])}</p><p class='caveat'>Single immutable historical run for catboost_snapshot. No second run is permitted. This result does not validate the preregistered v13 candidate.</p><div class='hash'>confirmation {file_sha256(c/'confirmation.json')} · lock {file_sha256(c/'confirmation.lock')} · model {cal['model_sha256']}</div></section>
<section id='preregistered' class='tier next'><div class='tag'>TIER 3 · PREREGISTERED NEXT STUDY · NO OUTCOMES ACCESSED</div><h2>{esc(cand['score'])}</h2><div class='grid'><div class='kpi'><span>Status</span><strong>{esc(pre['status'])}</strong></div><div class='kpi'><span>Calibration positives</span><strong>≥{ns['recommended_calibration_positive_events']}</strong></div><div class='kpi'><span>Evaluation positives</span><strong>≥{ns['recommended_evaluation_positive_events']}</strong></div><div class='kpi'><span>Threshold</span><strong>NOT SET</strong></div></div><p>{esc(cand['model'])}; hard fraction {cand['hard_fraction']}, hard mass {cand['hard_mass']}, {cand['iterations']} iterations, window {cand['decision_window_days'][0]}–{cand['decision_window_days'][1]} d, minimum history {cand['minimum_history']}.</p><h3>Prospective evaluation planning</h3><table><tr><th>Positive events</th><th>Max failures</th><th>UCB at max</th><th>P(pass | true danger=5%)</th></tr>{prow}</table><p class='caveat'>confirmation_v1 was previously unblinded and cannot be reused. v13 has calibration_accessed=false, evaluation_accessed=false, and threshold=null. A genuinely new, disjoint study must be frozen before outcomes are opened.</p><div class='hash'>preregistration {prehash} · planning {planhash} · v13 model {modelhash}</div></section>
<footer class='caveat'>Dataset: ESA Collision Avoidance Challenge, Zenodo 10.5281/zenodo.4463683, CC BY 4.0. Target is high final calculated collision probability, not collision occurrence. Event-level exchangeability is required; no operational guarantee is claimed under arbitrary shift.</footer></main></body></html>"""
    atomic_write(output,doc)
    return {'confirmation_passed':passed,'danger_ucb':float(m['danger_ucb']),'criterion':criterion,'development_pareto_methods':[x['method'] for x in v10['summary'] if x['pareto_frontier']],'preregistration_frozen':pre['status']=='frozen-before-new-data','next_study_candidate':cand['score'],'calibration_accessed':v13['calibration_accessed'],'evaluation_accessed':v13['evaluation_accessed'],'v13_threshold':v13['threshold']}

def main():
    p=argparse.ArgumentParser(description='Build the three-tier scientific evidence dashboard'); p.add_argument('--root',type=Path,default=ROOT); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); print(json.dumps(build_evidence_dashboard(a.root,a.output),indent=2))
if __name__=='__main__': main()
