"""Create the final comparison report and figures from saved measurements."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
source = HERE / 'final_comparison'
output = HERE.parent
figdir = HERE / 'figures'
if (output / 'COMPARISON_REPORT.md').exists():
    raise FileExistsError('The comparison report already exists; preserve earlier results')
figdir.mkdir(parents=True, exist_ok=True)
summary = json.loads((source / 'comparison_summary.json').read_text())
metrics, paired = summary['metrics'], summary['paired']
rows = pd.read_csv(source / 'comparison_per_trip.csv', dtype={'trip_id':str})

for partition in ('train','student'):
    q = rows[rows.partition == partition]
    systems = ['S1','S2','S3','S4']
    for metric, title in [('top1_all','Point top-1'),('continuity','Route continuity')]:
        fig, ax = plt.subplots(figsize=(7,4))
        vals = [q[q.system==s][metric].mean() for s in systems]
        ax.bar(systems,vals,color=['#4a75ad','#8d6eaa','#17a297','#bdbdbd'])
        ax.text(3, .06, 'NOT RUN', ha='center', va='bottom', rotation=90, fontsize=9, color='#555555')
        ax.set(title=f'{partition}: {title} by system', ylabel='Mean over trips', ylim=(0,1))
        fig.tight_layout(); fig.savefig(figdir/f'final_{partition}_{metric}.png',dpi=160); plt.close(fig)
    ref = q[q.system=='S1'].set_index('trip_id').sort_index()
    fig,ax=plt.subplots(figsize=(max(9,len(ref)*.14),4))
    for s in ('S2','S3'):
        v=q[q.system==s].set_index('trip_id').loc[ref.index]
        ax.plot(range(len(ref)),v.top1_all-ref.top1_all,'.',label=f'{s} − S1')
    ax.axhline(0,color='grey',lw=.7); ax.set(title=f'{partition}: paired trip top-1 differences',xlabel='Trip (sorted by ID)',ylabel='Top-1 difference'); ax.legend()
    fig.tight_layout(); fig.savefig(figdir/f'final_{partition}_paired.png',dpi=160); plt.close(fig)
    h=q[q.system=='S3'].set_index('trip_id').loc[ref.index]
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,col in zip(axes,['gap_share','far_share']):
        ax.scatter(h[col],h.top1_all-ref.top1_all,s=15); ax.axhline(0,color='grey',lw=.7)
        ax.set(xlabel=col.replace('_',' '),ylabel='S3 − S1 top-1')
    fig.suptitle(f'{partition}: HMM difference and trip characteristics')
    fig.tight_layout(); fig.savefig(figdir/f'final_{partition}_characteristics.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4)); ax.scatter(h.n_points,h.decode_seconds,s=18)
    ax.set(title=f'{partition}: HMM decode time by trip size',xlabel='GPS points',ylabel='Seconds')
    fig.tight_layout(); fig.savefig(figdir/f'final_{partition}_runtime.png',dpi=160); plt.close(fig)

def percent(x): return f'{100*x:.2f}%'
def stat(x): return f'{x:.4f}'
def verdict(interval):
    return 'real gain' if interval[0]>0 else ('worse' if interval[1]<0 else 'within noise')
lines = ['# Final model and route decoder comparison','',
 '## Executive summary','',
 '1. The final classifier is the untuned LightGBM.',
 '2. The original training-only decision found that the tuned gain was within noise.',
 '3. The final route decoder is the untuned-model HMM with `trained_models/hmm/config.json`.',
 f"4. On student-test, pooled all-point top-1 was {percent(metrics['student_S1']['top1_all'])} for the classifier and {percent(metrics['student_S3']['top1_all'])} with HMM.",
 f"5. Student mean-trip continuity was {percent(metrics['student_S1']['continuity_trip_mean'])} and {percent(metrics['student_S3']['continuity_trip_mean'])}, respectively.",
 f"6. Student mean-trip arc F1 against the reference route was {stat(metrics['student_S1']['arc_f1_vs_reference_trip_mean'])} and {stat(metrics['student_S3']['arc_f1_vs_reference_trip_mean'])}.",
 '7. The HMM improves continuity but lowers pooled point accuracy.',
 '8. The paired mean-trip student top-1 difference has a confidence interval that includes zero.',
 '9. The tuned classifier is shown for comparison; its HMM was not run.',
 '10. The student set has only ten trips; its results did not select the model or decoder.', '',
 '## How we got here','',
 'The tuning study selected an untuned final classifier because its prespecified training-only decision required a positive bootstrap interval for the in-candidate CV gain and no worse normalised log loss. The recorded gain interval was [−0.00149, +0.00269], so it included zero. The existing final-model directory already contains the untuned model. The HMM was selected using out-of-fold untuned-model probabilities and training trips; it was then evaluated on all training trips and the held-out student trips. No tuned-model HMM or new decoder tuning was run here.',
 '', '## Master comparison','',
 'CV columns are means and sample standard deviations across the five stored folds. Student top-1, macro F1 and log loss are pooled over points or candidate rows as appropriate; continuity, route F1 and mismatch are mean trip values. Route metrics are **vs reference route**. Times are measured in the original runs, not estimates of end-to-end latency.','',
 '| System | CV top-1 all | CV top-1 in candidates | Student top-1 all / in candidates / clean | Student candidate macro F1 | Student normalised log loss | Student continuity | Student reference route F1 | Student length mismatch | Time per 1,000 points | Final fit time |',
 '|---|---|---|---|---|---|---|---|---|---|---|']
for s,label,seconds,timing in [('S1','Untuned classifier',12.53,'3.395 ms model'),('S2','Tuned classifier',31.05,'16.609 ms model'),('S3','Untuned + HMM **FINAL**',12.53,f"{metrics['student_S3']['decode_ms_per_1000']:.2f} ms decode only")]:
    t,u=metrics['train_'+s],metrics['student_'+s]
    ll=stat(u['hard_decision_logloss_norm'])+' (hard)' if u['logloss_norm'] is None else stat(u['logloss_norm'])
    lines.append(f"| {s} {label} | {stat(t['cv_top1_all_mean'])} ± {stat(t['cv_top1_all_std'])} | {stat(t['cv_top1_in_candidates_mean'])} ± {stat(t['cv_top1_in_candidates_std'])} | {stat(u['top1_all'])} / {stat(u['top1_in_candidates'])} / {stat(u['top1_clean'])} | {stat(u['macro_f1'])} | {ll} | {stat(u['continuity_trip_mean'])} | {stat(u['arc_f1_vs_reference_trip_mean'])} | {stat(u['length_mismatch_vs_reference_trip_mean'])} | {timing} | {seconds:.2f} s classifier |")
lines += ['| S4 Tuned + HMM | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |','',
 'S4 was **NOT RUN: the tuned model did not beat the untuned model on cross-validation under the original decision rule**. The HMM returns a hard path, not posterior probabilities; a genuine posterior log loss is **NOT DEFINED**. The value marked “hard” is reconstructed for one-hot path decisions using the common metric’s probability clipping and normalisation. It is dominated by the clipping floor and is not comparable to a calibrated classifier loss. Candidate macro F1 is computed from the resulting one-arc-per-point confusion counts. The HMM decode timing excludes classifier inference and preprocessing. The reference length mismatch uses the multiset symmetric difference of arc lengths divided by reference-route length; it ignores order and can exceed one.',
 '', '## Detailed pooled results','',
 '| Partition / system | All top-1 | In-candidate top-1 | Clean top-1 | Candidate miss | Candidate macro F1 | Normalised log loss | Continuity (mean trip) | Broken transitions | Reference precision / recall / F1 (mean trip) | Reference length mismatch (mean trip) |',
 '|---|---:|---:|---:|---:|---:|---:|---:|---|---:|']
for p in ('train','student'):
    for s in ('S1','S2','S3'):
        v=metrics[p+'_'+s]; ll=stat(v['hard_decision_logloss_norm'])+' (hard)' if v['logloss_norm'] is None else stat(v['logloss_norm'])
        lines.append(f"| {p} {s} | {stat(v['top1_all'])} | {stat(v['top1_in_candidates'])} | {stat(v['top1_clean'])} | {stat(v['miss_rate'])} | {stat(v['macro_f1'])} | {ll} | {stat(v['continuity_trip_mean'])} | {v['broken_transitions']} | {stat(v['arc_precision_vs_reference_trip_mean'])} / {stat(v['arc_recall_vs_reference_trip_mean'])} / {stat(v['arc_f1_vs_reference_trip_mean'])} | {stat(v['length_mismatch_vs_reference_trip_mean'])} |")
for p in ('train','student'):
    v=metrics[p+'_S3']; largest=v['largest_trip']
    lines.append(f"\n{p} S3 changed {v['points_changed']} points, recorded {v['breaks']} breaks and relaxed {v['gap_steps']} gap steps. Decode time was {v['decode_seconds']:.2f} s total; the largest trip {largest['trip_id']} had {largest['points']} points and took {largest['seconds']:.3f} s.")
lines += ['', '## What each step added','',
 'The table uses a paired bootstrap of the **mean per-trip** top-1 difference; it is not the original decision-rule estimate, which used a different pooled estimand. A positive mean-trip interval in this descriptive table does not reopen the frozen classifier decision.','',
 '| Partition | Comparison | Mean trip top-1 difference | 95% paired trip bootstrap CI | Verdict |', '|---|---|---:|---|---|']
for p in ('train','student'):
    for label,key in [('Tuning (S2 − S1)','S2_minus_S1'),('HMM on untuned (S3 − S1)','S3_minus_S1')]:
        v=paired[p+'_'+key]['top1_all']; ci=v['ci95']
        lines.append(f"| {p} | {label} | {v['mean_trip_difference']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {verdict(ci)} |")
    lines.append(f'| {p} | HMM on tuned (S4 − S2) | NOT RUN | NOT RUN | NOT RUN |')
    lines.append(f'| {p} | Tuning + HMM (S4 − S1) | NOT RUN | NOT RUN | NOT RUN |')
lines += ['', '### Additional paired bootstrap intervals', '', '| Partition | Comparison | Metric | Mean trip difference | 95% CI |', '|---|---|---|---:|---|']
for p in ('train','student'):
    for key in ('S2_minus_S1','S3_minus_S1'):
        for metric in ('top1_in_candidates','continuity'):
            v = paired[p+'_'+key][metric]; ci = v['ci95']
            lines.append(f"| {p} | {key.replace('_minus_', ' − ')} | {metric} | {v['mean_trip_difference']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
lines += ['', 'For continuity, the paired mean-trip HMM difference and its bootstrap interval are:']
for p in ('train','student'):
    v=paired[p+'_S3_minus_S1']['continuity']; ci=v['ci95']
    lines.append(f"- {p}: {v['mean_trip_difference']:+.4f}, 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}] ({verdict(ci)}).")
lines += ['', '## Decoder settings','',
 '| Parameter | Untuned final config | Tuned-model config |','|---|---:|---|',
 '| alpha | 2 | NOT RUN |','| beta | 30 m | NOT RUN |','| U-turn penalty | 2 | NOT RUN |',
 '| gap beta factor | 3 | NOT RUN |','| minimum cutoff | 100 m | NOT RUN |','| top-k | 15 | NOT RUN |',
 '', 'These settings won the final seeded training-only decoder search on the selected trip subset. Earlier exploratory runs are preserved and excluded. There is no tuned-model config because Branch B applies.',
 '', '## Where the HMM helped and did not','',
 'Positive and negative differences below use all-point top-1 and are not causal diagnoses. The full trip table and weak-component diagnostics are linked below.']
for p in ('train','student'):
    q=rows[rows.partition==p]; base=q[q.system=='S1'].set_index('trip_id'); hmm=q[q.system=='S3'].set_index('trip_id').loc[base.index]
    delta=(hmm.top1_all-base.top1_all).sort_values()
    lines.append(f"\n{p}: {int((delta>0).sum())} trips improved, {int((delta<=0).sum())} were equal or worse. Largest gains: " + ', '.join(f'{i} {v:+.4f}' for i,v in delta.tail(3).iloc[::-1].items()) + '.')
    lines.append('Equal or worse: ' + ', '.join(f'{i} {v:+.4f}' for i,v in delta[delta<=0].items()) + '.')
lines += ['', 'For each non-improving trip, `trained_models/hmm/results/REPORT_ANSWERS.md` records observed gap, far-label, stationary, sparse-step, repeated-arc and road-component indicators where applicable. These are possible associations; aggregate indicators cannot prove a cause. Far-heavy trips can have reference-route coverage gaps. Sparse observations can skip arcs, so the collapsed per-point sequence can omit traversed arcs.',
 '', '## Caveats','',
 f"- Candidate misses set the shared all-point ceiling. The observed miss rates were {percent(metrics['train_S1']['miss_rate'])} on train and {percent(metrics['student_S1']['miss_rate'])} on student-test.",
 '- Ambiguous labels are more common in student-test (17.30%) than train (6.74%); far labels are concentrated in particular training trips.',
 '- Only LightGBM received this tuning; the other four model families were not tuned in the same way.',
 '- Student-test contains ten trips. The trip bootstrap is correspondingly uncertain, and continuity and point accuracy answer different questions.',
 '- The study export has 47 COMPLETE and three FAIL records, while the tuning narrative describes 46 complete evaluations and 41 unique configurations. This record-count discrepancy has not been resolved; the previously recorded decision was not recomputed.',
 '- The final and model4 untuned model files differ in line endings; their content matches after CRLF normalisation, which the backend model loader already performs.',
 '- The decoder requires a candidate at the first GPS point; it cannot carry an arc before one has been selected.',
 '', '## Figures','']
for p in ('train','student'):
    for metric,label in [('top1_all','Top-1 across systems'),('continuity','Continuity across systems'),('paired','Paired trip differences'),('characteristics','HMM differences and trip characteristics'),('runtime','Decode time versus trip size')]:
        lines.append(f'- [{p}: {label}](comparison/figures/final_{p}_{metric}.png)')
lines += ['', '[Per-trip comparison](comparison/final_comparison/comparison_per_trip.csv) · [Comparison summary and bootstrap intervals](comparison/final_comparison/comparison_summary.json) · [Earlier HMM trip diagnostics](hmm/results/REPORT_ANSWERS.md)',
 '', '## Bayesian optimisation questions','',
 '1. **What was searched?** Learning rate, leaves, depth, minimum leaf size, feature and bagging fractions, L1/L2 and split gain.',
 '2. **What ranges and why?** The search bracketed the untuned settings with practical values; full ranges are recorded in the tuning report.',
 '3. **What was the objective?** Mean in-candidate top-1 across the stored trip-grouped training folds, with normalised log loss used for near ties.',
 '4. **How many trials?** The narrative reports 41 unique completed configurations and 46 completed evaluations; the trial-export discrepancy is noted above.',
 '5. **What were the best values?** Trial 23 used learning rate 0.0239, 232 leaves, depth 12, minimum leaf size 205, feature fraction 0.543, bagging fraction 0.582, L1 0.002778, L2 0.2308 and split gain 0.3347. These were not the deployed settings.',
 '6. **Did tuning win?** No: the original paired interval included zero, so the untuned classifier remained final.',
 '', '## HMM questions','',
 '1. **Hidden states:** directed candidate road arcs at each GPS observation.',
 '2. **Emissions:** classifier probabilities normalised within each point, floored, logged and weighted.',
 '3. **Transitions:** the discrepancy between directed road distance and GPS displacement, plus a reverse-turn penalty.',
 '4. **Missing candidates and gaps:** carry the previous arc when possible; use real elapsed time and a relaxed transition scale for a gap.',
 '5. **Disconnected segments:** if every candidate pair is unreachable, record a break and restart; otherwise unreachable pairs get a very low score.',
 '6. **Did it improve the route?** Continuity and reference-route F1 increased, but pooled point accuracy fell; the student mean-trip top-1 difference was within the bootstrap interval around zero.',
 '', '## API hand-off','',
 'Set `MODEL_DIR` to the absolute path of `trained_models/final_model` (or `../trained_models/final_model` relative to the backend directory). Set `HMM_MODULE=trained_models.hmm.api_adapter:decode_trip` and put the repository root on `PYTHONPATH` before starting the backend. The adapter loads `trained_models/hmm/config.json`, the final untuned decoder configuration, and converts the backend network dictionary to the representation used during evaluation. It does not load reference routes or labels. No backend or frontend files were changed.',
 '', 'The previous direct `trained_models.hmm.hmm_decoder:decode_trip` setting alone is not compatible with the backend network dictionary, and the backend defaults do not load the frozen decoder settings. Use the adapter above. Re-check this hand-off with a live API smoke test in the deployment environment; this report verifies the adapter contract with unit tests but does not claim a full live-server test.', '']
(output/'COMPARISON_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print('Report and figures created')
