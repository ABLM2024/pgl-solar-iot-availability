"""Builds tables.json (main and supplementary tables) and vals.json from the result files."""
import json, os, numpy as np, pandas as pd
RS = '/home/claude/apr/res/'; os.makedirs(RS, exist_ok=True)
W = '/home/claude/apr/work/'
T = {}; V = {}
f1 = lambda x: f'{x:.1f}'; f3 = lambda x: f'{x:.3f}'; sg = lambda x, d=1: (f'+{x:.{d}f}' if x > 0 else f'{x:.{d}f}').replace('-', '−')
def ci(v, lo, hi, d=1): return f'{sg(v, d)} [{sg(lo, d)}, {sg(hi, d)}]'
# ---------------- Table 1 literature
lit = [
 ['Kumar et al. (2023)', 'Accuracy of low-cost PM2.5 data', 'None', 'Correction model; ARIMA–Kalman gap filling', 'PurpleAir and EPA sites, 2019–2021 (field)', 'n.r.', 'Missing data (~40%) discussed, not predicted'],
 ['McCarron et al. (2026)', 'Winter power-loss days of a solar-powered sensor', 'PVGIS energy simulation', 'None', 'Solar-powered PurpleAir set-up; simulation', 'n.a.', 'Outage fraction simulated; not linked to concentrations'],
 ['Zurita Macias and Trilles (2024)', 'Battery level of solar IoT devices', 'None', 'ML regression on meteorological variables', 'Deployed devices and weather records (field)', 'n.r.', 'Sampling-frequency adaptation'],
 ['Yamin and Bhat (2023)', 'Harvested energy with uncertainty', 'None', 'Low-power neural network', 'Five years of solar data, Golden, CO (historical)', 'n.r.', 'No'],
 ['Sarang et al. (2023)', 'Energy-harvesting rate for duty cycling', 'None', 'Nonlinear autoregressive network', '96 h of solar data in a network simulator', 'n.r.', 'No'],
 ['Gerndt et al. (2025)', 'Available solar energy, cloudy-day sequences', 'Not stated', 'Not stated', 'Low-cost solar board, case study', 'n.r.', 'No'],
 ['Dobrilović et al. (2023)', 'Solar-panel voltage of a sensor node', 'None', 'Nonlinear regression', 'Distributed WSN measurements (field)', 'n.r.', 'No'],
 ['Španer et al. (2025)', 'Next-day energy availability of an off-grid supply', 'Irradiance forecast and SoC monitoring', 'Predictive energy-management rules', 'Prototype, 2.5 kWh LiFePO4', 'n.r.', 'No'],
 ['Al-Jamal et al. (2026)', 'Residual energy of WSN nodes', 'None', 'TCN–LSTM with constrained scheduler', 'Real WSN energy and security dataset', 'n.r.', 'No'],
 ['Pollo et al. (2025)', 'Li-ion state of charge, variable horizons', 'Battery-dynamics equations (PINN)', 'Neural network', 'Two public battery datasets', 'Horizons', 'No'],
 ['This study', 'Daily reporting availability of each node (h = 1–28 d); bias of network PM2.5 estimators', 'VRLA energy balance, LVD, reconnection, operational latch, assimilation', 'Physics-prior residual boosting with monotone constraints; statistical and data-driven baselines', '20-node network, 366 days (field); two EEA reference points', 'Temporal (fixed, rolling, nested) and held-out nodes', 'Bias and exceedance capture of three estimators (reference proxy); data capture per node']]
T['lit'] = dict(caption='Table 1. Comparison of the ten most comparable studies identified by the literature search with the present study. Entries summarize the published abstracts and, where accessible, the full texts; “n.r.” means not reported there, “n.a.” not applicable.',
                head=['Study', 'Predicted quantity', 'Physics component', 'Learning component', 'Data', 'Generalization tests', 'Link to data representativeness'], rows=lit, hl=True)
# ---------------- Table 2 estimators
E = pd.read_csv(W + 'res_estimators.csv'); rows = []
def g(thr, lim, s): return E[(E.thr == thr) & (E.limit == lim) & (E.site == s)].iloc[0]
for thr in (0.25, 0.75):
    tl = '25%' if thr == 0.25 else '75%'
    for lab, bk, ek, dk in (('Daily network mean, ≥ 1 node', 'daily1_bias', 'daily1_exc', 'daily1_days'), ('Daily network mean, ≥ 2 nodes', 'daily2_bias', 'daily2_exc', 'daily2_days'),
                            ('Daily network mean, ≥ 5 nodes', 'daily5_bias', 'daily5_exc', 'daily5_days'), ('Pooled node-day mean, Eq. (1)', 'pooled_bias', 'pooled_exc', None),
                            ('Per-node annual means, median (range)', 'node_bias_median', 'node_exc_median', None)):
        r = [tl, lab]
        for s in 'AB':
            a = g(thr, 15.0, s); b = g(thr, 25.0, s)
            if bk == 'node_bias_median':
                bias = f"{sg(a[bk])} ({sg(a.node_bias_min)} to {sg(a.node_bias_max)})"
                if not np.isnan(a.get(bk + '_lo', np.nan)): bias += f" [{sg(a[bk + '_lo'])}, {sg(a[bk + '_hi'])}]"
            elif bk + '_lo' in a and not np.isnan(a[bk + '_lo']): bias = ci(a[bk], a[bk + '_lo'], a[bk + '_hi'])
            else: bias = sg(a[bk])
            days = str(int(a[dk])) if dk else str(int(a.n_days))
            r += [days, bias, f1(a[ek]), f1(b[ek])]
        rows.append([r[0], r[1], r[2] + '/' + r[6]] + r[3:6] + r[7:10])
T['est'] = dict(caption='Table 2. Temporal-coverage bias (%) of three estimators of the network PM2.5 mean under the reference-proxy assumption, with 95% weekly block-bootstrap intervals, and percentage of exceedance days sampled at 15 µg m⁻³ (98 and 86 days) and 25 µg m⁻³ (32 and 38 days), for activity thresholds of 25% and 75%. Days: reference days in the estimate at sites A/B; >15 and >25: percentage of exceedance days sampled. For per-node means the exceedance capture is the median over nodes; only nodes 1 and 18 met the 75% and 90% data-capture criteria at either threshold.',
               head=['Threshold', 'Estimator', 'Days A/B', 'Bias A (%)', '>15 A (%)', '>25 A (%)', 'Bias B (%)', '>15 B (%)', '>25 B (%)'], rows=rows, hl=True)
# ---------------- Table 3 decomposition and counterfactuals
Dd = pd.read_csv(W + 'res_decomp.csv'); Pn = pd.read_csv(W + 'res_panel.csv').set_index('Wp')
lab = {'Measured (active node-days / 20)': 'Measured availability (Eq. (1) basis)', 'Physics, as installed (10 Wp, open-loop, full-year calibration)': 'Calibrated physics as installed (10 Wp, open-loop)',
       'All commissioned nodes up': 'All commissioned nodes up (no outage of any cause)', 'Measured + restore maintenance-class down days': 'Measured, maintenance-class node-days restored',
       'Measured + restore other non-energy down days': 'Measured, other non-energy node-days restored', 'Measured + restore common-mode days': 'Measured, common-mode node-days restored',
       'Measured + restore energy-deficit down days': 'Measured, energy-deficit node-days restored', 'Measured + all non-energy down days restored': 'Measured, all non-energy node-days restored',
       'Physics counterfactual, 21 Wp': 'Calibrated physics, 21 Wp'}
rows = []
djf_meas = 11.7
for _, r in Dd.iterrows():
    dj = {'Measured (active node-days / 20)': f1(djf_meas), 'Physics, as installed (10 Wp, open-loop, full-year calibration)': f1(Pn.loc[10, 'djf_uptime']), 'All commissioned nodes up': '100.0',
          'Physics counterfactual, 21 Wp': f1(r.djf_uptime)}.get(r.source, '–')
    rows.append([lab[r.source], f1(r.mean_avail), dj, sg(r.bias_A), f1(r.exc_A), sg(r.bias_B), f1(r.exc_B)])
for wp in (16, 25, 50):
    r = Pn.loc[wp]; rows.append([f'Calibrated physics, {wp} Wp', f1(r.mean_avail), f1(r.djf_uptime), sg(r.bias_A), f1(r.exc_A), sg(r.bias_B), f1(r.exc_B)])
T['decomp'] = dict(caption='Table 3. Measured, simulated and counterfactual availability: mean network availability (20-node basis), December–February (DJF) uptime of commissioned node-days, bias (%) of the pooled estimator and percentage of days above 15 µg m⁻³ sampled. Restored classes are set to active; the maintenance class is an assumption (Section 2.5). Physics rows use factors calibrated on the full year.',
                   head=['Availability source', 'Mean availability (%)', 'DJF uptime (%)', 'Bias A (%)', 'Capture A (%)', 'Bias B (%)', 'Capture B (%)'], rows=rows, hl=True)
# ---------------- Table 4 forecasting
Pp = pd.read_csv(W + 'res3_pooled.csv'); P2 = pd.read_csv(W + 'res_roll_summary.csv'); R2 = pd.read_csv(W + 'res2_temporal_orig.csv') if os.path.exists(W + 'res2_temporal_orig.csv') else None
order = ['Persistence', 'Seasonal logistic (day of year)', 'Logistic lag model (Markov-type)', 'Discrete-time hazard model', 'Physics prior', 'ML data-driven', 'ML data-driven, tuned',
         'Concatenation, constrained', 'Concatenation, unconstrained', 'Residual on non-physical prior', 'Proposed', 'Proposed, tuned', 'Proposed without calendar features']
nice = {'ML data-driven': 'Data-driven ML', 'ML data-driven, tuned': 'Data-driven ML, tuned', 'Proposed': 'Proposed physics-guided learner', 'Proposed, tuned': 'Proposed, tuned',
        'Concatenation, constrained': 'Concatenation hybrid, constrained', 'Concatenation, unconstrained': 'Concatenation hybrid, unconstrained', 'Residual on non-physical prior': 'Residual learner, non-physical prior',
        'Physics prior': 'Physics prior, Eq. (12)'}
rows = []
def cell(split, h, m, k):
    q = Pp[(Pp.split == split) & (Pp.h == h) & (Pp.model == m)]; return q.iloc[0][k]
for m in order:
    r = [nice.get(m, m)]
    for h in (7, 14, 28): r += [f3(cell('roll', h, m, 'brier')), f1(cell('roll', h, m, 'mae'))]
    r += [f3(cell('roll', 28, m, 'brier_noApr')), f3(cell('fixed', 28, m, 'brier')), f3(cell('fixed', 28, m, 'auroc'))]
    rows.append(r)
# bold best per numeric column (min except AUROC)
for j in range(1, len(rows[0])):
    vals = [float(r[j]) for r in rows]; b = max(vals) if j == len(rows[0]) - 1 else min(vals)
    for r in rows:
        if float(r[j]) == b: r[j] = f'**{r[j]}**'
T['fc'] = dict(caption='Table 4. Forecast skill for node availability: rolling-origin Brier score (BS) and network-availability error MAE~net~ (pp), pooled over the eight test months (November 2025–June 2026), BS at 28 d without April, and BS and AUROC at 28 d in the fixed split (test January–June 2026). Bold: best value per column. All horizons and months are given in Supplementary Table S4.',
               head=['Model', 'BS 7 d', 'MAE 7 d', 'BS 14 d', 'MAE 14 d', 'BS 28 d', 'MAE 28 d', 'BS 28 d, no April', 'BS 28 d fixed', 'AUROC 28 d fixed'], rows=rows, hl=True)
# ---------------- Table 5 bias forecast
B = pd.read_csv(W + 'res_biasforecast.csv'); BP = pd.read_csv(W + 'res_biasforecast_vs_persistence.csv'); rows = []
for h in (7, 28):
    for m in ['Persistence', 'Climatology', 'Physics, assimilated', 'Physics prior', 'Data-driven ML', 'Concatenation', 'Proposed']:
        r = [f'{h} d', {'Physics, assimilated': 'Physics, assimilated (load state)', 'Concatenation': 'Concatenation hybrid', 'Proposed': 'Proposed physics-guided learner'}.get(m, m)]
        for s in 'AB':
            q = B[(B.h == h) & (B.model == m) & (B.site == s)].iloc[0]; r += [sg(q.bias_window_pred), ci(q.error, q.err_lo, q.err_hi)]
            v = BP[(BP.h == h) & (BP.model == m) & (BP.site == s)]
            r += [ci(v.iloc[0].diff_abs_err_vs_persistence, v.iloc[0].lo, v.iloc[0].hi) if len(v) else '–']
        rows.append(r)
mA = B.bias_window_meas[B.site == 'A'].iloc[0]; mB = B.bias_window_meas[B.site == 'B'].iloc[0]
T['biasfc'] = dict(caption=f'Table 5. Forecast of the bias of the pooled estimator over November 2025–June 2026 (measured: {sg(mA)}% at site A and {sg(mB)}% at site B): forecast bias (%), its error (forecast minus measured, pp) and the difference of its absolute error from that of persistence (pp; negative favours the model), with 95% weekly block-bootstrap intervals (unadjusted). All rows use the active-node basis of Eq. (1).',
                   head=['h', 'Model', 'Bias A (%)', 'Error A (pp)', '|Error| vs persistence A (pp)', 'Bias B (%)', 'Error B (pp)', '|Error| vs persistence B (pp)'], rows=rows, hl=True)
# ======================= Supplementary tables
import sys; sys.path.insert(0, W); os.chdir(W)
import feats2 as F
desc = {'act_t': 'active at origin t', 'a_t': 'sample fraction at t', 'act_m7': 'active fraction, last 7 d', 'act_m14': 'active fraction, last 14 d', 'run_len': 'run length of current state (d)',
        'net_t': 'network active fraction at t', 'net_m7': 'network active fraction, last 7 d', 'net_chg7': 'change of network fraction over 7 d', 'dT_t': 'daily enclosure-temperature range at t',
        'dT_m7': '7-d mean range', 'Tmean_t': 'daily mean enclosure temperature', 'Tmin_t': 'daily minimum enclosure temperature', 'Tmean_m7': '7-d mean temperature', 'ntemp_t': 'number of temperature-reporting nodes',
        'doy_sin': 'sine of day of year of target', 'doy_cos': 'cosine of day of year of target', 'm_pole': 'dedicated pole (0/1)', 'm_lamp': 'lamp pole (0/1)', 'm_bldg': 'building bracket (0/1)', 'age': 'days since commissioning',
        's_n': 'effective energy-balance factor', 'soc_t': 'assimilated state of charge at t', 'on_t': 'assimilated load state at t', 'off_days_t': 'days disconnected at t', 'latched_t': 'latch at t',
        'soc_h': 'propagated state of charge at t + h', 'soc_min': 'minimum propagated state of charge', 'up_sim': 'simulated load state at t + h', 'ndown_sim': 'simulated down days to t + h',
        'cum_net': 'cumulative climatological net balance', 'Hclim_h': 'climatological irradiation at t + h', 'net_clim_h': 'climatological net balance at t + h', 'cap_h': 'capacity factor at t + h', 'acc_h': 'charge-acceptance factor at t + h'}
rows = [[f, 'data' if f in F.DATA_F else 'physics', desc[f], {1: '+1', -1: '−1'}.get(F.MONO.get(f, 0), 'none'), 'yes' if f in ('doy_sin', 'doy_cos', 'age') else ''] for f in F.DATA_F + F.PHYS_F]
pf = ['up_sim', 'soc_h', 'soc_min', 'on_t', 'latched_t']
for r in rows: r.append('yes' if r[0] in pf else '')
T['S1'] = dict(caption='Table S1. Features of the learners with monotone signs κ~j~ (physics features only) and membership of the calendar group and of the physics prior (Eq. (12)). The data-driven learner uses the data features; the cold-start variant omits the five own-history features.',
               head=['Feature', 'Group', 'Description', 'Monotone sign', 'Calendar', 'In prior'], rows=rows)
N = pd.read_csv(W + 'res_nodecapture.csv')
T['S2'] = dict(caption='Table S2. Data capture of each node over its commissioned period and over the calendar period of the study, bias of its annual mean (%) and percentage of exceedance days (> 15 µg m⁻³) sampled, reference-proxy assumption, 25% activity threshold.',
               head=['Node', 'Mounting', 'Commissioned', 'Capture, commissioned (%)', 'Capture, calendar (%)', 'Bias A (%)', 'Capture >15 A (%)', 'Bias B (%)', 'Capture >15 B (%)', '≥ 75%', '≥ 90%'],
               rows=[[int(r.node), r.mount, r.commissioned, f1(r.capture_pct), f1(r.capture_cal_pct), sg(r.bias_A), f1(r.exc_A), sg(r.bias_B), f1(r.exc_B), 'yes' if r.meets75 else 'no', 'yes' if r.meets90 else 'no'] for _, r in N.iterrows()])
PB = pd.read_csv(W + 'res_proxy_bounds.csv')
T['S3'] = dict(caption='Table S3. Reference-proxy bounds: bias (%) of the pooled, daily (≥ 1 node) and median per-node estimators when the nodes are assumed to measure the reference concentration modified by the stated error structure; the target is the mean over commissioned node-days. Assumed scenarios, not measurements.',
               head=['Site', 'Scenario', 'Pooled (%)', 'Daily, ≥ 1 node (%)', 'Per-node median (%)'], rows=[[r.site, r.scenario, sg(r.pooled), sg(r.daily1), sg(r.node_median)] for _, r in PB.iterrows()])
R3 = pd.read_csv(W + 'res3_temporal.csv'); R2o = pd.read_csv(W + 'res2_temporal.csv')
NM2 = {'Climatology': 'Climatology', 'Physics v1 open-loop (s=1)': 'Physics, original', 'Physics v2 open-loop (calibrated)': 'Physics, calibrated open-loop', 'Physics v2 assimilated': 'Physics, assimilated (load state)'}
extra = R2o[R2o.model.isin(NM2)].copy(); extra['model'] = extra.model.map(NM2)
RR = pd.concat([R3, extra[extra.h.isin([7, 14, 28])]]); RR = RR[RR.split.str.startswith('roll')]
rows = []
for (m, h), q in RR.groupby(['model', 'h']):
    q = q.set_index('split').sort_index(); rows.append([nice.get(m, m), int(h)] + [f'{b:.3f} / {e:.1f}' for b, e in zip(q.brier, q.net_mae_pp)])
months = sorted(RR.split.unique())
T['S4'] = dict(caption='Table S4. Rolling-origin results per test month: Brier score / network-availability error (pp) at 7, 14 and 28 d for all models (h = 1 d in the deposited res2_temporal.csv).',
               head=['Model', 'h (d)'] + [pd.Timestamp(m[5:] + '-01').strftime('%b %Y') for m in months], rows=rows)
BA = pd.read_csv(W + 'res_boot_all.csv')
T['S5'] = dict(caption='Table S5. Differences between the proposed learner and the baselines (proposed minus baseline; negative favours the proposed learner) with 95% weekly block-bootstrap intervals (4,000 resamples), two-sided bootstrap p-values and Holm-adjusted p-values within each family. Family 1: 48 pre-defined comparisons of the proposed learner with the physics prior, the data-driven learner and persistence; family 2: comparisons with the additional statistical, tuned and ablation baselines; family 3: hazard model minus data-driven learner.',
               head=['Family', 'Design', 'h (d)', 'Baseline', 'Metric', 'Difference [95% CI]', 'p', 'p (Holm)'],
               rows=[[int(r.family), r.split, int(r.h), nice.get(r.vs, r.vs), 'BS' if r.metric == 'brier' else 'MAE~net~ (pp)', ci(r['diff'], r.lo, r.hi, 3 if r.metric == 'brier' else 1), f'{r.p:.4f}', f'{r.p_holm:.3f}'] for _, r in BA.iterrows()])
S = pd.read_csv(W + 'res_s_profile.csv')
T['S9'] = dict(caption='Table S9. Effective energy-balance factors s~n~: best value, maximum balanced accuracy and near-optimal range (within 0.01 of the maximum) for calibration on June–December 2025 (train) and on the full year; censored values reach the edge of the grid 0.3–3.0.',
               head=['Node', 'Mounting', 'Period', 's best', 'Balanced accuracy', 'Near-optimal range', 'Censored'],
               rows=[[int(r.node), r.mount, r.period, f'{r.s_best:.1f}', f'{r.ba_max:.3f}', f'{r.near_lo:.1f}–{r.near_hi:.1f}', 'upper' if r.censored_upper else ('lower' if r.censored_lower else '')] for _, r in S.iterrows()])
BD = pd.read_csv(W + 'res_brierdecomp.csv'); EV = pd.read_csv(W + 'res_events.csv'); LT = pd.read_csv(W + 'res_leadtime.csv')
rows = []
for _, r in BD.iterrows():
    e = EV[(EV.split == r.split) & (EV.h == r.h) & (EV.model == r.model)].set_index('event')
    def ef(ev, k): v = e.loc[ev, k]; return '–' if pd.isna(v) else f'{v:.2f}'
    rows.append([r.split, int(r.h), nice.get(r.model, r.model), f'{r.brier:.4f}', f'{r.reliability:.4f}', f'{r.resolution:.4f}', f'{r.uncertainty:.4f}', ef('failure', 'POD'), ef('failure', 'false_alarm_rate'), ef('failure', 'precision'), ef('restart', 'POD'), ef('restart', 'false_alarm_rate')])
T['S10'] = dict(caption='Table S10. Brier decomposition (reliability, resolution, uncertainty; Murphy, 1973; 10 probability bins) and event metrics at a threshold of 0.5: probability of detection (POD), false-alarm rate (FAR) and precision for failures (node active at the origin, down at the target) and restarts (down at the origin, active at the target). The reliability diagrams are given numerically in res_reliability.csv. Lead time (rolling design, 45 first-down days after an active day): largest horizon at which the failure was predicted (p < 0.5): ' +
                '; '.join(f"{nice.get(r.model, r.model)}: none {r.lead_0}, 1 d {r.lead_1}, 7 d {r.lead_7}, 14 d {r.lead_14}, 28 d {r.lead_28}" for _, r in LT.iterrows()) + '.',
                head=['Design', 'h (d)', 'Model', 'BS', 'Reliability', 'Resolution', 'Uncertainty', 'POD failure', 'FAR failure', 'Precision failure', 'POD restart', 'FAR restart'], rows=rows)
SH = pd.read_csv(W + 'res_shap.csv'); DG = pd.read_csv(W + 'res_dec_diag.csv')
T['S11'] = dict(caption='Table S11. TreeSHAP attributions of the proposed learner (mean contribution to the logit and mean absolute contribution, averaged over five seeds) by feature group and for the leading features, and diagnosis of the folds: training and test active fractions, mean physics prior, mean prediction, simulated load state and propagated state of charge. ' +
                ' '.join(f"{r.fold}: train {r.train_pos:.2f}, test {r.test_pos:.2f}, prior {r.prior_mean:.2f}, prediction {r.proposed_mean:.2f}, simulated on {r.up_sim_mean:.2f}, state of charge at t + h {r.soc_h_mean:.2f}." for _, r in DG.iterrows()),
                head=['Fold', 'Group or feature', 'Mean contribution (logit)', 'Mean |contribution|'], rows=[[r.fold, r.family.replace('top: ', 'feature: '), sg(r.mean_contrib, 3), f'{r.mean_abs_contrib:.3f}'] for _, r in SH.iterrows()])
MF = pd.read_csv(W + 'res_biasforecast_monthly.csv')
rows = []
for (h, m, s), q in MF.groupby(['h', 'model', 'site']):
    q = q.set_index('month').sort_index(); rows.append([int(h), m, s] + [sg(x) for x in q.err])
T['S13'] = dict(caption='Table S13. Monthly error (pp) of the forecast bias of the pooled estimator (forecast minus measured bias within each month).',
                head=['h (d)', 'Model', 'Site'] + [pd.Timestamp(m + '-01').strftime('%b %Y') for m in sorted(MF.month.unique())], rows=rows)
PN = pd.read_csv(W + 'res_panel.csv'); PNN = pd.read_csv(W + 'res_panel_node.csv'); PU = pd.read_csv(W + 'res_panel_unc.csv')
rows = [[f'{int(r.Wp)} Wp', 'sweep', f1(r.djf_uptime), sg(r.bias_A), f1(r.exc_A), sg(r.bias_B), f1(r.exc_B), f1(r.djf_uptime_nolatch)] for _, r in PN.iterrows()]
rows += [['21 Wp', r.case, f1(r.djf_uptime), sg(r.bias_A), f1(r.exc_A), sg(r.bias_B), f1(r.exc_B), '–'] for _, r in PU.iterrows()]
T['S14'] = dict(caption='Table S14. Panel-size counterfactual in the calibrated model: DJF uptime of commissioned node-days (%), pooled bias (%) and exceedance capture (%) with the latch, and DJF uptime without the latch; uncertainty of the 21-Wp case (near-optimal factors, five best structures, winter irradiation ±15%, assumed). Panel for 99% DJF uptime per node (latch / no latch, Wp): ' +
                ', '.join(f"node {int(r.node)} {int(r.Wp99_latch)}" for _, r in PNN.iterrows()) + f' (median {PNN.Wp99_latch.median():.0f} Wp; identical without the latch).',
                head=['Panel', 'Case', 'DJF uptime (%)', 'Bias A (%)', 'Capture A (%)', 'Bias B (%)', 'Capture B (%)', 'DJF uptime, no latch (%)'], rows=rows)
TU = pd.read_csv(W + 'res3_tuning.csv')
T['S15'] = dict(caption='Table S15. Hyperparameters selected by nested time-series validation (last 28 training days; grid n ∈ {100, 300, 600} trees × {7, 15, 31} leaves).',
                head=['Fold', 'h (d)', 'Data-driven: trees', 'Data-driven: leaves', 'Proposed: trees', 'Proposed: leaves'], rows=[[r.split, int(r.h), int(r.dd_n), int(r.dd_leaves), int(r.pr_n), int(r.pr_leaves)] for _, r in TU.iterrows()])
# pending tables are added by tables_rev2.py
json.dump(T, open(RS + 'tables.json', 'w'), ensure_ascii=False, indent=0)
print('tables', list(T))
