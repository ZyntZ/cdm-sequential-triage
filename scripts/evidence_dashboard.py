"""Build a self-contained dashboard separating development, confirmation, and preregistered evidence."""
from __future__ import annotations
import argparse, html, json, os, tempfile
from pathlib import Path
from typing import Any
import pandas as pd
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from confirmation import file_sha256, read_json, validate_calibration_artifact

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


def _svg_text(value: Any) -> str:
    return html.escape(str(value), quote=True)


def frontier_svg(summary: list[dict[str, Any]], alpha: float) -> str:
    """Render the safety-automation frontier as a self-contained SVG."""
    if not summary:
        raise ValueError("Frontier summary must not be empty")
    width, height = 760, 390
    left, right, top, bottom = 78, 30, 28, 62
    plot_w, plot_h = width-left-right, height-top-bottom
    x_values=[float(row['danger_ucb']) for row in summary]
    y_values=[float(row['safe_negative_rate']) for row in summary]
    x_min=min(min(x_values),alpha)-0.005; x_max=max(max(x_values),alpha)+0.005
    y_min=min(y_values)-0.015; y_max=max(y_values)+0.015
    if x_max<=x_min or y_max<=y_min: raise ValueError('Degenerate frontier range')
    sx=lambda value: left+(float(value)-x_min)/(x_max-x_min)*plot_w
    sy=lambda value: top+(y_max-float(value))/(y_max-y_min)*plot_h
    parts=[f"<svg class='evidence-plot' viewBox='0 0 {width} {height}' role='img' aria-labelledby='frontier-title frontier-desc'>",
           "<title id='frontier-title'>Development safety-automation frontier</title>",
           "<desc id='frontier-desc'>Danger upper confidence bound on the horizontal axis and safe-negative automation rate on the vertical axis. The dashed vertical line is the ten percent development criterion.</desc>",
           f"<rect x='{left}' y='{top}' width='{plot_w}' height='{plot_h}' class='plot-bg'/>"]
    for i in range(5):
        x=x_min+(x_max-x_min)*i/4; px=sx(x)
        parts.append(f"<line x1='{px:.1f}' y1='{top}' x2='{px:.1f}' y2='{top+plot_h}' class='gridline'/><text x='{px:.1f}' y='{height-34}' text-anchor='middle' class='axis-label'>{100*x:.1f}%</text>")
        y=y_min+(y_max-y_min)*i/4; py=sy(y)
        parts.append(f"<line x1='{left}' y1='{py:.1f}' x2='{left+plot_w}' y2='{py:.1f}' class='gridline'/><text x='{left-10}' y='{py+4:.1f}' text-anchor='end' class='axis-label'>{100*y:.1f}%</text>")
    criterion_x=sx(alpha)
    parts.append(f"<line x1='{criterion_x:.1f}' y1='{top}' x2='{criterion_x:.1f}' y2='{top+plot_h}' class='criterion'/><text x='{criterion_x+6:.1f}' y='{top+15}' class='criterion-label'>UCB criterion {100*alpha:.0f}%</text>")
    for row in summary:
        px,py=sx(row['danger_ucb']),sy(row['safe_negative_rate'])
        classes=['point']
        if row.get('pareto_frontier'): classes.append('pareto')
        if not row.get('safety_feasible'): classes.append('infeasible')
        method=_svg_text(row['method'])
        parts.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='7' class='{' '.join(classes)}'><title>{method}: UCB {100*float(row['danger_ucb']):.2f}%, automation {100*float(row['safe_negative_rate']):.2f}%</title></circle><text x='{px+10:.1f}' y='{py-9:.1f}' class='point-label'>{method}</text>")
    parts.extend([f"<text x='{left+plot_w/2:.1f}' y='{height-7}' text-anchor='middle' class='axis-title'>Dangerous-exclusion UCB95 (lower is safer)</text>",
                  f"<text x='18' y='{top+plot_h/2:.1f}' text-anchor='middle' transform='rotate(-90 18 {top+plot_h/2:.1f})' class='axis-title'>Correct SAFE-EXCLUDE rate (higher is better)</text>","</svg>"])
    return ''.join(parts)


def fold_stability_svg(folds: pd.DataFrame) -> str:
    """Render paired candidate-minus-baseline coverage changes by outer fold."""
    required={'method','fold','coverage_delta','coverage_delta_ci_low','coverage_delta_ci_high'}
    missing=required.difference(folds.columns)
    if missing: raise ValueError(f"Missing fold columns: {sorted(missing)}")
    selected=folds.loc[folds['method'].eq('catboost_tail_aligned')].sort_values('fold')
    if selected.empty: raise ValueError('No catboost_tail_aligned fold rows')
    width,height=760,270; left,right,top,bottom=78,28,24,42
    plot_w=width-left-right; row_h=(height-top-bottom)/len(selected)
    low=min(-0.035,float(selected['coverage_delta_ci_low'].min())-0.005)
    high=max(0.08,float(selected['coverage_delta_ci_high'].max())+0.005)
    sx=lambda value: left+(float(value)-low)/(high-low)*plot_w
    zero=sx(0)
    parts=[f"<svg class='evidence-plot' viewBox='0 0 {width} {height}' role='img' aria-labelledby='fold-title fold-desc'>",
           "<title id='fold-title'>Outer-fold coverage stability</title>",
           "<desc id='fold-desc'>Paired change in correct SAFE-EXCLUDE rate for the tail-aligned candidate versus the snapshot baseline. Confidence intervals crossing neither side of zero indicate a consistent directional change within that fold.</desc>",
           f"<line x1='{zero:.1f}' y1='{top-5}' x2='{zero:.1f}' y2='{height-bottom+3}' class='zero-line'/>"]
    for index,row in enumerate(selected.itertuples(index=False)):
        y=top+(index+0.5)*row_h; x1=sx(row.coverage_delta_ci_low); x2=sx(row.coverage_delta_ci_high); x=sx(row.coverage_delta)
        cls='gain' if row.coverage_delta>0 else 'loss'
        parts.append(f"<text x='{left-12}' y='{y+4:.1f}' text-anchor='end' class='axis-label'>fold {int(row.fold)}</text><line x1='{x1:.1f}' y1='{y:.1f}' x2='{x2:.1f}' y2='{y:.1f}' class='ci-line'/><line x1='{x1:.1f}' y1='{y-5:.1f}' x2='{x1:.1f}' y2='{y+5:.1f}' class='ci-line'/><line x1='{x2:.1f}' y1='{y-5:.1f}' x2='{x2:.1f}' y2='{y+5:.1f}' class='ci-line'/><circle cx='{x:.1f}' cy='{y:.1f}' r='7' class='fold-point {cls}'/><text x='{x+11:.1f}' y='{y+4:.1f}' class='point-label'>{100*row.coverage_delta:+.2f} pp</text>")
    for value in (-0.02,0,0.02,0.04,0.06,0.08):
        if low<=value<=high:
            x=sx(value); parts.append(f"<text x='{x:.1f}' y='{height-12}' text-anchor='middle' class='axis-label'>{100*value:+.0f}</text>")
    parts.append(f"<text x='{left+plot_w/2:.1f}' y='{height-1}' text-anchor='middle' class='axis-title'>Coverage change, percentage points</text></svg>")
    return ''.join(parts)


SUPPORTED_LOCALES = {"en", "ru"}


def localize_evidence_document(document: str, locale: str) -> str:
    if locale not in SUPPORTED_LOCALES:
        raise ValueError(f"Unsupported locale: {locale}")
    if locale == "en":
        return document
    replacements = [
        ("<html>", "<html lang='ru'>"),
        ("CDM Triage Evidence", "Доказательная панель триажа CDM"),
        ("SCIENTIFIC EVIDENCE RECORD · NOT FOR OPERATIONS", "НАУЧНАЯ ДЕМОНСТРАЦИЯ · НЕ ДЛЯ ЭКСПЛУАТАЦИИ"),
        ("Sequential CDM Triage · Evidence Dashboard", "Последовательный триаж CDM · Доказательная панель"),
        ("TIER 1 · EXPLORATORY DEVELOPMENT EVIDENCE", "УРОВЕНЬ 1 · ИССЛЕДОВАТЕЛЬСКИЕ DEVELOPMENT-РЕЗУЛЬТАТЫ"),
        ("Safety–automation frontier", "Граница безопасность–автоматизация"),
        ("Development safety-automation frontier", "Development-граница безопасность–автоматизация"),
        ("Danger upper confidence bound on the horizontal axis and safe-negative automation rate on the vertical axis. The dashed vertical line is the ten percent development criterion.", "По горизонтали показана верхняя доверительная граница опасного исключения, по вертикали — доля корректной автоматизации. Пунктирная линия соответствует development-критерию 10%."),
        ("UCB criterion", "Критерий UCB"),
        ("Dangerous-exclusion UCB95 (lower is safer)", "UCB95 опасного исключения (меньше — безопаснее)"),
        ("Correct SAFE-EXCLUDE rate (higher is better)", "Доля корректных SAFE-EXCLUDE (больше — лучше)"),
        ("Selected development candidate:", "Выбранный development-кандидат:"),
        ("tail-aligned CatBoost retained", "tail-aligned CatBoost сохранил"),
        ("dangerous exclusions", "опасных исключений"),
        ("and increased correct SAFE-EXCLUDE coverage by", "и увеличил долю корректных SAFE-EXCLUDE на"),
        ("percentage points", "процентного пункта"),
        ("exact McNemar", "точный критерий Мак-Немара"),
        ("Method", "Метод"), ("Dangerous exclusions", "Опасные исключения"),
        ("<th>Danger</th>", "<th>Опасные</th>"),
        ("Safe-negative", "Корректные исключения"),
        ("Feasible", "Допустим"), ("Pareto", "Парето"),
        (">YES<", ">ДА<"), (">NO<", ">НЕТ<"),
        ("Outer-fold stability of the selected candidate", "Устойчивость выбранного кандидата по outer folds"),
        ("Outer-fold coverage stability", "Устойчивость покрытия по outer folds"),
        ("Paired change in correct SAFE-EXCLUDE rate for the tail-aligned candidate versus the snapshot baseline. Confidence intervals crossing neither side of zero indicate a consistent directional change within that fold.", "Парное изменение доли корректных SAFE-EXCLUDE для tail-aligned кандидата относительно snapshot baseline. Доверительные интервалы показывают неопределённость внутри каждого fold."),
        ("Coverage change, percentage points", "Изменение покрытия, процентные пункты"),
        ("The pooled automation gain is heterogeneous: folds 1–3 improve, while folds 0 and 4 decline. This is reported as a limitation and motivates the genuinely new validation study; no subgroup-specific safety guarantee is claimed.", "Объединённый прирост автоматизации неоднороден: folds 1–3 улучшаются, а folds 0 и 4 ухудшаются. Это явно указано как ограничение и обосновывает новую независимую проверку; гарантия безопасности для отдельных подгрупп не заявляется."),
        ("250 correlated calibration-split repeats", "250 зависимых повторов разделения calibration/test"),
        ("Repeats", "Повторы"), ("Median coverage", "Медианное покрытие"),
        ("Coverage gain &gt; 0", "Прирост покрытия &gt; 0"), ("Danger not worse", "Опасные не хуже"),
        ("Development data only: 7,146 events, 192 positives. Fixed out-of-fold scores were reused across correlated calibration splits; these are not independent retraining replications and are not confirmation evidence.", "Только development-данные: 7 146 событий, 192 положительных. Фиксированные out-of-fold scores повторно использовались в зависимых разделениях calibration/test; это не независимые переобучения и не подтверждающее доказательство."),
        ("TIER 2 · LOCKED CONFIRMATION_V1 · CRITERION NOT MET", "УРОВЕНЬ 2 · ЗАМОРОЖЕННЫЙ CONFIRMATION_V1 · КРИТЕРИЙ НЕ ВЫПОЛНЕН"),
        ("TIER 2 · LOCKED CONFIRMATION_V1 · CRITERION MET", "УРОВЕНЬ 2 · ЗАМОРОЖЕННЫЙ CONFIRMATION_V1 · КРИТЕРИЙ ВЫПОЛНЕН"),
        ("Snapshot-model confirmation", "Подтверждение snapshot-модели"),
        ("PRE-SPECIFIED CRITERION NOT MET", "ЗАРАНЕЕ ЗАДАННЫЙ КРИТЕРИЙ НЕ ВЫПОЛНЕН"),
        ("PRE-SPECIFIED CRITERION MET", "ЗАРАНЕЕ ЗАДАННЫЙ КРИТЕРИЙ ВЫПОЛНЕН"),
        ("Observed danger", "Наблюдаемая доля"),
        ("Median lead", "Медианное упреждение"),
        ("calibration bound", "граница calibration"),
        ("Single immutable historical run for catboost_snapshot. No second run is permitted. This result does not validate the preregistered v13 candidate.", "Единственный неизменяемый исторический запуск catboost_snapshot. Повторный запуск не допускается. Этот результат не подтверждает предзарегистрированный кандидат v13."),
        ("TIER 3 · PREREGISTERED NEXT STUDY · NO OUTCOMES ACCESSED", "УРОВЕНЬ 3 · ПРЕДРЕГИСТРАЦИЯ СЛЕДУЮЩЕГО ИССЛЕДОВАНИЯ · ИСХОДЫ НЕ ОТКРЫТЫ"),
        ("Status", "Статус"), ("Calibration positives", "Положительные calibration"),
        ("Evaluation positives", "Положительные evaluation"), ("Threshold", "Порог"),
        ("NOT SET", "НЕ УСТАНОВЛЕН"), ("iterations", "итераций"),
        ("window", "окно"), ("minimum history", "минимальная история"),
        ("Prospective evaluation planning", "Планирование prospective evaluation"),
        ("Positive events", "Положительные события"), ("Max failures", "Макс. ошибок"),
        ("UCB at max", "UCB при максимуме"),
        ("confirmation_v1 was previously unblinded and cannot be reused. v13 has calibration_accessed=false, evaluation_accessed=false, and threshold=null. A genuinely new, disjoint study must be frozen before outcomes are opened.", "Исходы confirmation_v1 уже были открыты, поэтому повторное использование невозможно. Для v13: calibration_accessed=false, evaluation_accessed=false, threshold=null. До открытия исходов необходимо заморозить новое непересекающееся исследование."),
        ("Dataset:", "Набор данных:"),
        ("Target is high final calculated collision probability, not collision occurrence. Event-level exchangeability is required; no operational guarantee is claimed under arbitrary shift.", "Целью является высокая финальная расчётная вероятность столкновения, а не факт столкновения. Требуется обменность событий; эксплуатационная гарантия при произвольном сдвиге распределения не заявляется."),
    ]
    for source, target in replacements:
        document = document.replace(source, target)
    return document

def build_evidence_dashboard(root:Path,output:Path,locale:str="en")->dict[str,Any]:
    if locale not in SUPPORTED_LOCALES: raise ValueError(f"Unsupported locale: {locale}")
    if output.exists(): raise FileExistsError(f"Evidence dashboard exists: {output}")
    a=root/'artifacts'; r=root/'reports'; c=a/'confirmation_v1'
    v10=read_json(a/'development_score_ensemble_v10.json'); v11=read_json(a/'development_score_ensemble_repeated_v11.json')
    conf=read_json(c/'confirmation.json'); cal=read_json(c/'calibration.json'); clock=read_json(c/'confirmation.lock')
    pre=read_json(a/'next_validation_preregistration_v12.json'); plock=read_json(a/'next_validation_preregistration_v12.lock')
    v13=read_json(a/'catboost_tail_aligned_final_v13.json'); plan=pd.read_csv(r/'next_validation_sample_size_v12.csv'); folds=pd.read_csv(r/'development_score_ensemble_folds_v10.csv')
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
    validate_calibration_artifact(cal, conf.get('policy'))
    if v13['threshold'] is not None or v13['calibration_accessed'] is not False or v13['evaluation_accessed'] is not False: raise ValueError('v13 is not frozen before calibration')
    criterion=float(conf['policy']['alpha']); passed=float(conf['evaluation']['danger_ucb'])<=criterion
    devrows=''.join(f"<tr><td>{esc(x['method'])}</td><td>{x['danger_k']}/{x['danger_n']}</td><td>{pct(x['danger_ucb'])}</td><td>{pct(x['safe_negative_rate'])}</td><td>{'YES' if x['safety_feasible'] else 'NO'}</td><td>{'YES' if x['pareto_frontier'] else 'NO'}</td></tr>" for x in v10['summary'])
    stab=''.join(f"<tr><td>{esc(x['method'])}</td><td>{int(x['repeats'])}</td><td>{pct(x['median_safe_negative_rate'])}</td><td>{pct(x['coverage_delta_positive_fraction'])}</td><td>{pct(x['danger_not_worse_fraction'])}</td></tr>" for x in v11['summary'])
    pc='pass_probability_if_true_rate_0.05'; prow=''.join(f"<tr><td>{int(x['positive_events'])}</td><td>{int(x['maximum_passing_failures'])}</td><td>{pct(x['upper_bound_at_maximum'])}</td><td>{pct(x[pc])}</td></tr>" for _,x in plan.iterrows())
    m=conf['evaluation']; cr=cal['calibration']; cand=pre['candidate']; ns=pre['new_study']; frontier=frontier_svg(v10['summary'],float(pre['candidate']['alpha'])); foldplot=fold_stability_svg(folds); tail=next(x for x in v10['summary'] if x['method']=='catboost_tail_aligned')
    css="body{margin:0;background:#071018;color:#e6eef2;font:14px Arial}main{max-width:1350px;margin:auto;padding:28px}h1{margin:5px 0 20px}.tier{border:1px solid #263b48;border-left:5px solid;padding:20px;margin:18px 0;background:#0d1b25}.dev{border-left-color:#efb84b}.conf{border-left-color:#ff6375}.next{border-left-color:#48c7df}.tag{font:700 12px monospace;letter-spacing:.12em}.dev .tag{color:#efb84b}.conf .tag{color:#ff6375}.next .tag{color:#48c7df}.fail{background:#35151d;border:1px solid #ff6375;padding:12px;font-weight:bold}table{width:100%;border-collapse:collapse;margin:12px 0}th,td{padding:8px;border-bottom:1px solid #213541;text-align:left}th{color:#91a8b5;font:11px monospace}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.kpi{background:#09151e;padding:12px}.kpi strong,.kpi span{display:block}.kpi strong{font-size:22px}.caveat{color:#afbec6;border-left:3px solid #788c97;padding-left:10px}.hash{font:11px monospace;color:#8fa5b2;word-break:break-all}.evidence-plot{width:100%;height:auto;background:#09151e;border:1px solid #213541;margin:12px 0}.plot-bg{fill:#09151e}.gridline{stroke:#203541;stroke-width:1}.axis-label,.point-label,.criterion-label{fill:#afbec6;font:12px Arial}.axis-title{fill:#d5e2e8;font:13px Arial}.criterion{stroke:#ff6375;stroke-width:2;stroke-dasharray:7 5}.criterion-label{fill:#ff8290}.point{fill:#788c97;stroke:#e6eef2;stroke-width:1}.point.pareto{fill:#48c7df;stroke:#c9f7ff;stroke-width:2}.point.infeasible{fill:#ff6375}.zero-line{stroke:#91a8b5;stroke-width:2}.ci-line{stroke:#afbec6;stroke-width:2}.fold-point{stroke:#e6eef2;stroke-width:1}.fold-point.gain{fill:#48c7df}.fold-point.loss{fill:#ff6375}.finding{background:#102632;border:1px solid #315064;padding:12px;margin:12px 0}@media(max-width:800px){.grid{grid-template-columns:1fr 1fr}}"
    doc=f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>CDM Triage Evidence</title><style>{css}</style></head><body><main><div class='tag'>SCIENTIFIC EVIDENCE RECORD · NOT FOR OPERATIONS</div><h1>Sequential CDM Triage · Evidence Dashboard</h1>
<section id='development' class='tier dev'><div class='tag'>TIER 1 · EXPLORATORY DEVELOPMENT EVIDENCE</div><h2>Safety–automation frontier</h2>{frontier}<div class='finding'><strong>Selected development candidate:</strong> tail-aligned CatBoost retained 12/192 dangerous exclusions (UCB95 {pct(tail['danger_ucb'])}) and increased correct SAFE-EXCLUDE coverage by {100*tail['coverage_delta']:.2f} percentage points (95% CI {100*tail['coverage_delta_ci_low']:.2f} to {100*tail['coverage_delta_ci_high']:.2f}; exact McNemar p={tail['coverage_mcnemar_p']:.3g}).</div><table><tr><th>Method</th><th>Danger</th><th>UCB95</th><th>Safe-negative</th><th>Feasible</th><th>Pareto</th></tr>{devrows}</table><h3>Outer-fold stability of the selected candidate</h3>{foldplot}<p class='caveat'>The pooled automation gain is heterogeneous: folds 1–3 improve, while folds 0 and 4 decline. This is reported as a limitation and motivates the genuinely new validation study; no subgroup-specific safety guarantee is claimed.</p><h3>250 correlated calibration-split repeats</h3><table><tr><th>Method</th><th>Repeats</th><th>Median coverage</th><th>Coverage gain &gt; 0</th><th>Danger not worse</th></tr>{stab}</table><p class='caveat'>Development data only: 7,146 events, 192 positives. Fixed out-of-fold scores were reused across correlated calibration splits; these are not independent retraining replications and are not confirmation evidence.</p><div class='hash'>v10 {file_sha256(a/'development_score_ensemble_v10.json')} · v11 {file_sha256(a/'development_score_ensemble_repeated_v11.json')}</div></section>
<section id='confirmation' class='tier conf'><div class='tag'>TIER 2 · LOCKED CONFIRMATION_V1 · {'CRITERION MET' if passed else 'CRITERION NOT MET'}</div><h2>Snapshot-model confirmation</h2><div class='fail'>PRE-SPECIFIED CRITERION {'MET' if passed else 'NOT MET'}: UCB {pct(m['danger_ucb'])} {'≤' if passed else '>'} α {pct(criterion)}.</div><div class='grid'><div class='kpi'><span>Dangerous exclusions</span><strong>{m['danger_k']}/{m['danger_n']}</strong></div><div class='kpi'><span>Observed danger</span><strong>{pct(m['danger_rate'])}</strong></div><div class='kpi'><span>Safe-negative</span><strong>{pct(m['safe_negative_rate'])}</strong></div><div class='kpi'><span>Median lead</span><strong>{m['median_first_safe_tca']:.2f} d</strong></div></div><p>PAC rank {cr['rank']}/{cr['n_positive']} · threshold {cr['threshold']:.8g} · calibration bound {pct(cr['pac_bound'])}</p><p class='caveat'>Single immutable historical run for catboost_snapshot. No second run is permitted. This result does not validate the preregistered v13 candidate.</p><div class='hash'>confirmation {file_sha256(c/'confirmation.json')} · lock {file_sha256(c/'confirmation.lock')} · model {cal['model_sha256']}</div></section>
<section id='preregistered' class='tier next'><div class='tag'>TIER 3 · PREREGISTERED NEXT STUDY · NO OUTCOMES ACCESSED</div><h2>{esc(cand['score'])}</h2><div class='grid'><div class='kpi'><span>Status</span><strong>{esc(pre['status'])}</strong></div><div class='kpi'><span>Calibration positives</span><strong>≥{ns['recommended_calibration_positive_events']}</strong></div><div class='kpi'><span>Evaluation positives</span><strong>≥{ns['recommended_evaluation_positive_events']}</strong></div><div class='kpi'><span>Threshold</span><strong>NOT SET</strong></div></div><p>{esc(cand['model'])}; hard fraction {cand['hard_fraction']}, hard mass {cand['hard_mass']}, {cand['iterations']} iterations, window {cand['decision_window_days'][0]}–{cand['decision_window_days'][1]} d, minimum history {cand['minimum_history']}.</p><h3>Prospective evaluation planning</h3><table><tr><th>Positive events</th><th>Max failures</th><th>UCB at max</th><th>P(pass | true danger=5%)</th></tr>{prow}</table><p class='caveat'>confirmation_v1 was previously unblinded and cannot be reused. v13 has calibration_accessed=false, evaluation_accessed=false, and threshold=null. A genuinely new, disjoint study must be frozen before outcomes are opened.</p><div class='hash'>preregistration {prehash} · planning {planhash} · v13 model {modelhash}</div></section>
<footer class='caveat'>Dataset: ESA Collision Avoidance Challenge, Zenodo 10.5281/zenodo.4463683, CC BY 4.0. Target is high final calculated collision probability, not collision occurrence. Event-level exchangeability is required; no operational guarantee is claimed under arbitrary shift.</footer></main></body></html>"""
    doc=localize_evidence_document(doc,locale)
    atomic_write(output,doc)
    return {'confirmation_passed':passed,'danger_ucb':float(m['danger_ucb']),'criterion':criterion,'development_pareto_methods':[x['method'] for x in v10['summary'] if x['pareto_frontier']],'preregistration_frozen':pre['status']=='frozen-before-new-data','next_study_candidate':cand['score'],'calibration_accessed':v13['calibration_accessed'],'evaluation_accessed':v13['evaluation_accessed'],'v13_threshold':v13['threshold'],'locale':locale}

def main():
    p=argparse.ArgumentParser(description='Build the three-tier scientific evidence dashboard'); p.add_argument('--root',type=Path,default=ROOT); p.add_argument('--output',type=Path,required=True); p.add_argument('--locale',choices=sorted(SUPPORTED_LOCALES),default='en'); a=p.parse_args(); print(json.dumps(build_evidence_dashboard(a.root,a.output,locale=a.locale),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
