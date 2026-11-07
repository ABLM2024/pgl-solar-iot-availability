"""Adds tables S6, S7, S8, S12 and the pending values to tables.json / vals.json."""
import json, os, numpy as np, pandas as pd
RS = '/home/claude/apr/res/'; W = '/home/claude/apr/work/'; os.chdir(W)
T = json.load(open(RS + 'tables.json')); V = {}
sg = lambda x, d=1: (f'+{x:.{d}f}' if x > 0 else f'{x:.{d}f}').replace('-', '−')
def ci(v, lo, hi, d=3): return f'{sg(v, d)} [{sg(lo, d)}, {sg(hi, d)}]'
# ---- S6 spatial
S = pd.read_csv('res2_spatial_all.csv')
NM = {'Hybrid: physics-prior residual (proposed)': 'Proposed', 'ML data-driven': 'Data-driven ML', 'Physics v2 calibrated prob.': 'Physics prior', 'Hybrid: feature concatenation': 'Concatenation hybrid',
      'Network persistence (others at origin)': 'Network persistence', 'Physics v2 assimilated': 'Physics, assimilated', 'Physics v2 open-loop': 'Physics, open-loop', 'Persistence': 'Persistence'}
SET = {'Spatio-temporal, own calibration': 'Held-out node, own s_n', 'LONO-spatiotemporal': 'Held-out node, median s_n', 'Cold-start spatio-temporal': 'Held-out node, no own history',
       'LONO-history': 'Concurrent leave-one-node-out (interpolation upper bound)', 'Cold-start': 'Concurrent, cold start (upper bound)'}
rows = []
for st in SET:
    for m in NM:
        q = S[(S.setting == st) & (S.model == m)]
        if not len(q): continue
        r = [SET[st], NM[m]]
        for h in (7, 28):
            x = q[q.h == h]; r += [f'{x.brier.iloc[0]:.3f}', f'{x.auroc.iloc[0]:.3f}'] if len(x) else ['–', '–']
        x = q[q.h == 28]; r.append(f'{x.net_mae_pp.iloc[0]:.1f}' if len(x) else '–'); rows.append(r)
BS = pd.read_csv('res_boot_spatial.csv')
bcap = ' Bootstrap of the proposed learner minus the baseline (held-out designs; B = 2,000): ' + '; '.join(
    f"{'own' if r.setting == 'own' else 'median'} s_n, {int(r.h)} d, vs {NM.get(r.vs, r.vs)}, {'BS' if r.metric == 'brier' else 'MAE'}: {sg(r['diff'], 3 if r.metric == 'brier' else 1)}, weekly [{sg(r.week_lo, 3 if r.metric == 'brier' else 1)}, {sg(r.week_hi, 3 if r.metric == 'brier' else 1)}], node-cluster [{sg(r.node_lo, 3 if r.metric == 'brier' else 1)}, {sg(r.node_hi, 3 if r.metric == 'brier' else 1)}], two-way [{sg(r['two-way_lo'], 3 if r.metric == 'brier' else 1)}, {sg(r['two-way_hi'], 3 if r.metric == 'brier' else 1)}], Holm-adjusted p (node-cluster) {r.node_p_holm:.2f}"
    for _, r in BS.iterrows() if r.h != 1) + '.'
T['S6'] = dict(caption='Table S6. Held-out nodes: Brier score (BS), AUROC and network-availability error at 28 d (pp). Held-out designs train on the other 19 nodes up to 31 December 2025 and test the held-out node from 1 January 2026; the concurrent designs use the other nodes over the whole year (their labels on the same target days) and are interpolation upper bounds, not forecasts.' + bcap,
               head=['Design', 'Model', 'BS 7 d', 'AUROC 7 d', 'BS 28 d', 'AUROC 28 d', 'MAE 28 d (pp)'], rows=rows)
r7 = BS[(BS.setting == 'own') & (BS.h == 7) & (BS.vs == 'ML data-driven') & (BS.metric == 'brier')].iloc[0]
V['sp7_week'] = f"[{sg(r7.week_lo, 3)}, {sg(r7.week_hi, 3)}]"; V['sp7_node'] = f"[{sg(r7.node_lo, 3)}, {sg(r7.node_hi, 3)}]"; V['sp7_two'] = f"[{sg(r7['two-way_lo'], 3)}, {sg(r7['two-way_hi'], 3)}]"
V['sp_holm'] = f"its adjusted p-value was {r7.node_p_holm:.2f} (node-cluster) and {r7['two-way_p_holm']:.2f} (two-way)"
# ---- S7 grid and nested
G = pd.read_csv('res_structgrid_fixed.csv'); NS = pd.read_csv('res_nested.csv')
rows = [[f'{r.dod:g}', f'{r.R:g}', '∞' if r.N_L == 999 else str(int(r.N_L)), f'{r.T_L:g}', f'{r.bacc_train:.3f}', f'{r.bacc_test_openloop:.3f}'] for _, r in G.head(25).iterrows()]
st = NS.drop_duplicates('split')[['split', 'dod', 'R', 'N_L', 'T_L', 'bacc_sel']]
def lab(r): return f"δ = {r.dod:g}, R = {r.R:g}, N_L = {'∞' if r.N_L == 999 else int(r.N_L)}, T_L = {r.T_L:g} °C"
T['S7'] = dict(caption='Table S7. Extended structure grid on June–December 2025 (25 best of ' + str(len(G)) + ' configurations, ranked by the training balanced accuracy of the open-loop simulation, with the open-loop balanced accuracy on January–June 2026). Structures selected inside each fold: ' + '; '.join(f"{r.split}: {lab(r)} (balanced accuracy {r.bacc_sel:.3f})" for _, r in st.iterrows()) +
               '. Scores with fold-specific structures (Brier score, network error, pp): ' + '; '.join(f"{m}, {h} d, rolling: {NS[(NS.model == m) & (NS.h == h) & NS.split.str.startswith('roll')].brier.mean():.3f} (monthly mean)" for m in NS.model.unique() for h in (7, 14, 28)) + '.',
               head=['δ', 'R', 'N_L (d)', 'T_L (°C)', 'BA train', 'BA test (open-loop)'], rows=rows)
b0 = G.iloc[0]; base = G[(G.dod == 0.9) & (G.R == 0.5) & (G.N_L == 3) & (G.T_L == 5.0)].iloc[0]
V['grid_best'] = f"best configuration {lab(b0)} with {b0.bacc_train:.3f}, against {base.bacc_train:.3f} for the original selection"
V['nested_structs'] = '; '.join(f"{r.split.replace('roll_', '')}: {lab(r)}" for _, r in st.iterrows())
# pooled nested 28-d rolling
import glob
def pooled(pat, col):
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pat))]); return np.mean((df[col] - df.y) ** 2)
V['nested_prop28'] = f"{pooled('p_nested_roll_*_h28.csv', 'proposed'):.3f}"; V['nested_prior28'] = f"{pooled('p_nested_roll_*_h28.csv', 'prior'):.3f}"
V['nested_prop14'] = f"{pooled('p_nested_roll_*_h14.csv', 'proposed'):.3f}"
# ---- S8 sensitivity
SE = pd.read_csv('res_sens.csv'); rows = []
LAB = {'base': 'Base (T_L 5 °C, ρ 0.2, PR 0.65, N_L 3, δ 0.9)', 'TL0': 'T_L = 0 °C', 'TL2.5': 'T_L = 2.5 °C', 'TL7.5': 'T_L = 7.5 °C', 'TL10': 'T_L = 10 °C', 'RES0.1': 'ρ = 0.1', 'RES0.3': 'ρ = 0.3',
       'PR0.55': 'PR = 0.55', 'PR0.75': 'PR = 0.75', 'NL1': 'N_L = 1 d', 'NLinf': 'N_L = ∞ (no latch)', 'DOD1.0': 'δ = 1.0', 'CMoff': 'Common-mode rule off', 'CMdec': 'Common-mode rule + 6–8 Dec 2025'}
out = {}
for l in LAB:
    q = SE[SE.label == l]
    if not len(q): continue
    rr = {}
    for m in ('Physics prior', 'Proposed'):
        x = q[(q.model == m) & (q.split == 'fixed')].iloc[0]; rr[m] = (x.brier, x.net_mae_pp)
    out[l] = (q.bacc_train.iloc[0], q.bacc_test_openloop.iloc[0], rr)
    rows.append([LAB[l], f'{q.bacc_train.iloc[0]:.3f}', f'{q.bacc_test_openloop.iloc[0]:.3f}', f"{rr['Physics prior'][0]:.3f}", f"{rr['Proposed'][0]:.3f}", f"{rr['Proposed'][1]:.1f}"])
T['S8'] = dict(caption='Table S8. Sensitivity of the physics and of the 28-d forecasts to the fixed parameters and the common-mode rule (one parameter changed at a time, factors recalibrated): open-loop balanced accuracy (BA) on June–December 2025 and January–June 2026, fixed-split 28-d Brier score of the physics prior and the proposed learner and network error of the proposed learner (pp). Because of computing time, the forecasts were evaluated on the fixed split only.',
               head=['Variant', 'BA train', 'BA test', 'BS prior, fixed 28 d', 'BS proposed, fixed 28 d', 'MAE proposed (pp)'], rows=rows)
ba = [v[0] for v in out.values()]; br = [v[2]['Proposed'][0] for v in out.values()]
V['sens_ba_range'] = f'{min(ba):.3f} to {max(ba):.3f}'; V['sens_br_range'] = f'{min(br):.3f} to {max(br):.3f} (base {out["base"][2]["Proposed"][0]:.3f})'
best = min(out, key=lambda l: out[l][2]['Proposed'][0]); worst = max(out, key=lambda l: out[l][2]['Proposed'][0])
V['sens_text'] = (f"the lowest fixed-split score was obtained with {LAB[best]} and the highest with {LAB[worst]}; " +
                  f"without the latch the training balanced accuracy fell to {out['NLinf'][0]:.3f} and the fixed-split 28-d Brier score of the proposed learner was {out['NLinf'][2]['Proposed'][0]:.3f}, and switching the common-mode rule off or extending it changed the score to {out['CMoff'][2]['Proposed'][0]:.3f} and {out['CMdec'][2]['Proposed'][0]:.3f}. For comparison, the data-driven learner had 0.110 and the hazard model 0.120 in the fixed split."
                  if 'NLinf' in out and 'CMoff' in out and 'CMdec' in out else '')
# ---- S12 75% threshold
P75 = pd.read_csv('res3_pooled_thr75.csv'); P25 = pd.read_csv('res3_pooled.csv'); rows = []
for m in P75.model.unique():
    r = [m]
    for P_ in (P25, P75):
        for sp, h in (('roll', 14), ('roll', 28), ('fixed', 28)): r.append(f"{P_[(P_.split == sp) & (P_.h == h) & (P_.model == m)].brier.iloc[0]:.3f}")
    rows.append(r)
T['S12'] = dict(caption='Table S12. Forecast skill (pooled Brier score) with activity thresholds of 25% (main) and 75%.', head=['Model', '25%: rolling 14 d', '25%: rolling 28 d', '25%: fixed 28 d', '75%: rolling 14 d', '75%: rolling 28 d', '75%: fixed 28 d'], rows=rows)
g = lambda m, sp='roll', h=28: P75[(P75.split == sp) & (P75.h == h) & (P75.model == m)].brier.iloc[0]
V['thr75_text'] = f"{g('Proposed'):.3f} for the proposed learner, {g('ML data-driven'):.3f} for the data-driven learner, {g('Discrete-time hazard model'):.3f} for the hazard model and {g('Persistence'):.3f} for persistence, so the hazard model remained best and the proposed learner was no better than the data-driven learner"
json.dump(T, open(RS + 'tables.json', 'w'), ensure_ascii=False, indent=0); json.dump(V, open(RS + 'vals.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(V, ensure_ascii=False, indent=1))
# ---- nested-structure rows in Table 4 and family-4 bootstrap values
from sklearn.metrics import roc_auc_score
fc = T['fc']; rows = [[c.replace('**', '') for c in r] for r in fc['rows']]
for col, name in (('prior', 'Physics prior, nested structure'), ('proposed', 'Proposed, nested structure')):
    r = [name]
    for h in (7, 14, 28):
        df = pd.concat([pd.read_csv(f, parse_dates=['date']) for f in sorted(glob.glob(f'p_nested_roll_*_h{h}.csv'))]); g = df.groupby('date').agg(y=('y', 'mean'), p=(col, 'mean'))
        r += [f'{np.mean((df[col] - df.y) ** 2):.3f}', f'{100 * np.abs(g.p - g.y).mean():.1f}']
    na = df[df.date.dt.month != 4]; r.append(f'{np.mean((na[col] - na.y) ** 2):.3f}')
    fx = pd.read_csv('p_nested_fixed_h28.csv'); r += [f'{np.mean((fx[col] - fx.y) ** 2):.3f}', f'{roc_auc_score(fx.y, fx[col]):.3f}']
    rows.append(r)
for j in range(1, len(rows[0])):
    vals = [float(r[j]) for r in rows]; b = max(vals) if j == len(rows[0]) - 1 else min(vals)
    for r in rows:
        if float(r[j]) == b: r[j] = f'**{r[j]}**'
fc['rows'] = rows
fc['caption'] = fc['caption'].replace('Bold: best value per column.', 'Nested structure: physics structure selected inside each fold on the extended grid (Section 2.7). Bold: best value per column.')
BN = pd.read_csv('res_boot_nested.csv')
def bn(h, vs, met='brier'):
    r = BN[(BN.h == h) & (BN.vs == vs) & (BN.metric == met)].iloc[0]; d = 3 if met == 'brier' else 1
    return f"{sg(r['diff'], d)} [{sg(r.lo, d)}, {sg(r.hi, d)}], adjusted p = {r.p_holm:.3f}"
V['nest_dd28'] = bn(28, 'ML data-driven'); V['nest_hz28'] = bn(28, 'Discrete-time hazard model'); V['nest_pers28'] = bn(28, 'Persistence'); V['nest_dd28_net'] = bn(28, 'ML data-driven', 'net')
V['nest_fixed28'] = f"{np.mean((fx.proposed - fx.y) ** 2):.3f}"
json.dump(T, open(RS + 'tables.json', 'w'), ensure_ascii=False, indent=0); json.dump(V, open(RS + 'vals.json', 'w'), ensure_ascii=False, indent=1)
print(V['nest_dd28'], V['nest_hz28'], V['nest_fixed28']); [print(r) for r in rows[-3:]]
