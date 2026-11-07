"""Analyses on per-prediction files -> res_*.csv"""
import sys, glob, numpy as np, pandas as pd
sys.path.insert(0, '.'); import core
d = core.load(); days = d['days']; D = core.D
ref = pd.read_csv(D + 'C_reference/reference_PM25_daily_EEA.csv', parse_dates=['date']).set_index('date').reindex(days)
SITES = {'A': 'PM25_siteA_SPO_RO0065A', 'B': 'PM25_siteB_SPO_RO0229A'}
A_rec = (d['act'] * d['valid']).sum(1) / 20
NAMES = {'Persistence': 'Persistence', 'Climatology': 'Climatology', 'Physics v1 open-loop (s=1)': 'Physics, original', 'Physics v2 open-loop (calibrated)': 'Physics, calibrated open-loop',
         'Physics v2 assimilated': 'Physics, assimilated', 'Physics v2 assimilated (calibrated prob.)': 'Physics prior', 'ML data-driven': 'Data-driven ML',
         'Hybrid: feature concatenation': 'Concatenation', 'Hybrid: physics-prior residual (proposed)': 'Proposed', 'Hybrid residual without monotone constraints': 'Proposed, unconstrained'}
def load(split, h, pat='p2_temporal'):
    fs = sorted(glob.glob(f'{pat}_{split}*_h{h}.csv')); return pd.concat([pd.read_csv(f, parse_dates=['date']) for f in fs]).rename(columns=NAMES) if fs else None
# ---- rolling results, all horizons, monthly and pooled aggregation
R = pd.read_csv('res2_temporal.csv'); R['model'] = R.model.map(NAMES)
roll = R[R.split.str.startswith('roll')]
mon = roll.pivot_table(index=['model', 'h'], columns='split', values=['brier', 'net_mae_pp']).round(3); mon.to_csv('res_roll_monthly.csv')
rows = []
for h in (1, 7, 14, 28):
    df = load('roll', h)
    for m in NAMES.values():
        if m not in df: continue
        g = df.groupby('date').agg(y=('y', 'mean'), p=(m, 'mean'))
        r = roll[(roll.model == m) & (roll.h == h)]
        ex = r[r.split != 'roll_2026-04']
        dfx = df[df.date.dt.month != 4]; gx = dfx.groupby('date').agg(y=('y', 'mean'), p=(m, 'mean'))
        rows.append(dict(model=m, h=h, brier_monthmean=r.brier.mean(), brier_pooled=np.mean((df[m] - df.y) ** 2), mae_monthmean=r.net_mae_pp.mean(),
                         mae_pooled=100 * np.abs(g.p - g.y).mean(), brier_pooled_noApr=np.mean((dfx[m] - dfx.y) ** 2), mae_pooled_noApr=100 * np.abs(gx.p - gx.y).mean(),
                         brier_dec=r[r.split == 'roll_2025-12'].brier.mean(), mae_dec=r[r.split == 'roll_2025-12'].net_mae_pp.mean(),
                         brier_apr=r[r.split == 'roll_2026-04'].brier.mean(), mae_apr=r[r.split == 'roll_2026-04'].net_mae_pp.mean()))
pd.DataFrame(rows).to_csv('res_roll_summary.csv', index=False)
print(pd.DataFrame(rows)[lambda x: x.model.isin(['Persistence', 'Physics prior', 'Data-driven ML', 'Proposed'])].round(3).to_string())
# ---- block bootstrap (pooled aggregation), Holm-adjusted p-values from the bootstrap distribution
def boot(df, a, b, metric, B=4000, seed=0, cluster='week'):
    df = df.copy(); df['wk'] = df.date.dt.to_period('W').astype(str)
    if metric == 'brier':
        df['ea'] = (df[a] - df.y) ** 2; df['eb'] = (df[b] - df.y) ** 2; unit = df
    else:
        unit = df.groupby('date').agg(y=('y', 'mean'), pa=(a, 'mean'), pb=(b, 'mean')).reset_index(); unit['wk'] = unit.date.dt.to_period('W').astype(str)
        unit['ea'] = 100 * np.abs(unit.pa - unit.y); unit['eb'] = 100 * np.abs(unit.pb - unit.y)
    g = unit.groupby('wk')[['ea', 'eb']].agg(['sum', 'count']); sa = g[('ea', 'sum')].values; sb = g[('eb', 'sum')].values; nn = g[('ea', 'count')].values
    est = (sa.sum() - sb.sum()) / nn.sum(); rng = np.random.default_rng(seed); W = len(nn); bs = np.empty(B)
    for i in range(B):
        j = rng.integers(0, W, W); bs[i] = (sa[j].sum() - sb[j].sum()) / nn[j].sum()
    lo, hi = np.percentile(bs, [2.5, 97.5]); p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return est, lo, hi, p, W
rows = []
for split in ('fixed', 'roll'):
    for h in (1, 7, 14, 28):
        df = load(split, h)
        for b in ('Physics prior', 'Persistence', 'Data-driven ML'):
            for met in ('brier', 'net'):
                e, lo, hi, p, W = boot(df, 'Proposed', b, met); rows.append(dict(split=split, h=h, vs=b, metric=met, diff=e, lo=lo, hi=hi, p=p, weeks=W))
B = pd.DataFrame(rows)
def holm(p):
    o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj
B['p_holm'] = holm(B.p.values); B.to_csv('res_boot_temporal.csv', index=False); print(B.round(4).to_string())
# ---- bias forecasts by model, horizon and month, with weekly block-bootstrap intervals of the error
def wbias(a, c):
    m = ~np.isnan(c) & ~np.isnan(a); return 100 * ((a[m] * c[m]).sum() / a[m].sum() - c[m].mean()) / c[m].mean()
rows = []; mrows = []
for h in (7, 28):
    df = load('roll', h); win = days >= pd.Timestamp('2025-11-01')
    for m in ['Persistence', 'Climatology', 'Physics prior', 'Physics, assimilated', 'Data-driven ML', 'Concatenation', 'Proposed']:
        g = (df.groupby('date')[m].sum() / 20).reindex(days).values
        for s, col in SITES.items():
            c = ref[col].values; meas = wbias(A_rec[win], c[win]); pred = wbias(g[win], c[win]); full = np.where(np.isnan(g), A_rec, g)
            # weekly block bootstrap of the forecast error of the window bias
            wk = pd.Series(days[win]).dt.to_period('W').astype(str).values; uw = np.unique(wk); idx = {w: np.where(wk == w)[0] for w in uw}; rng = np.random.default_rng(0); errs = []
            gw = g[win]; aw = A_rec[win]; cw = c[win]
            for _ in range(2000):
                sel = np.concatenate([idx[w] for w in rng.choice(uw, len(uw))]); errs.append(wbias(gw[sel], cw[sel]) - wbias(aw[sel], cw[sel]))
            lo, hi = np.percentile(errs, [2.5, 97.5])
            rows.append(dict(h=h, model=m, site=s, bias_window_meas=meas, bias_window_pred=pred, error=pred - meas, err_lo=lo, err_hi=hi, bias_full_pred=wbias(full, c), bias_full_meas=wbias(A_rec, c)))
            for M0 in pd.date_range('2025-11-01', '2026-06-01', freq='MS'):
                mm = (days >= M0) & (days <= M0 + pd.offsets.MonthEnd(0))
                mrows.append(dict(h=h, model=m, site=s, month=M0.strftime('%Y-%m'), err=wbias(g[mm], c[mm]) - wbias(A_rec[mm], c[mm]), mean_avail_pred=100 * np.nanmean(g[mm]), mean_avail_meas=100 * A_rec[mm].mean()))
BF = pd.DataFrame(rows); BF.to_csv('res_biasforecast.csv', index=False); MF = pd.DataFrame(mrows); MF.to_csv('res_biasforecast_monthly.csv', index=False)
print(BF.round(1).to_string())
# bootstrap: |error| of model vs |error| of persistence for the window bias (paired)
rows = []
for h in (7, 28):
    df = load('roll', h); win = days >= pd.Timestamp('2025-11-01'); wk = pd.Series(days[win]).dt.to_period('W').astype(str).values; uw = np.unique(wk); idx = {w: np.where(wk == w)[0] for w in uw}
    G = {m: (df.groupby('date')[m].sum() / 20).reindex(days).values[win] for m in ['Persistence', 'Physics prior', 'Data-driven ML', 'Proposed']}
    for s, col in SITES.items():
        cw = ref[col].values[win]; aw = A_rec[win]
        for m in ['Physics prior', 'Data-driven ML', 'Proposed']:
            rng = np.random.default_rng(0); dd = []
            for _ in range(2000):
                sel = np.concatenate([idx[w] for w in rng.choice(uw, len(uw))]); t = wbias(aw[sel], cw[sel])
                dd.append(abs(wbias(G[m][sel], cw[sel]) - t) - abs(wbias(G['Persistence'][sel], cw[sel]) - t))
            e = abs(wbias(G[m], cw) - wbias(aw, cw)) - abs(wbias(G['Persistence'], cw) - wbias(aw, cw)); lo, hi = np.percentile(dd, [2.5, 97.5])
            rows.append(dict(h=h, site=s, model=m, diff_abs_err_vs_persistence=e, lo=lo, hi=hi))
pd.DataFrame(rows).to_csv('res_biasforecast_vs_persistence.csv', index=False); print(pd.DataFrame(rows).round(1).to_string())
# ---- Brier decomposition, reliability curves, event metrics
def murphy(p, y, bins=10):
    b = np.clip((p * bins).astype(int), 0, bins - 1); ybar = y.mean(); rel = res = 0.0; curve = []
    for i in range(bins):
        m = b == i
        if m.any(): pk = p[m].mean(); ok = y[m].mean(); rel += m.sum() * (pk - ok) ** 2; res += m.sum() * (ok - ybar) ** 2; curve.append((i, m.sum(), pk, ok))
    return rel / len(p), res / len(p), ybar * (1 - ybar), curve
rows = []; crow = []; erow = []
for split in ('fixed', 'roll'):
    for h in (7, 28):
        df = load(split, h)
        for m in ['Persistence', 'Physics prior', 'Data-driven ML', 'Concatenation', 'Proposed']:
            p = np.clip(df[m].values, 0, 1); y = df.y.values; rel, res, unc, curve = murphy(p, y)
            rows.append(dict(split=split, h=h, model=m, brier=np.mean((p - y) ** 2), reliability=rel, resolution=res, uncertainty=unc))
            for i, nb, pk, ok in curve: crow.append(dict(split=split, h=h, model=m, bin=i, n=nb, p_mean=pk, obs=ok))
            # events: failures (active at origin, down at target) and restarts (down at origin, active at target)
            a0 = df.act_t.values if 'act_t' in df else None
            per = df.Persistence.values
            for ev, cond, hit in (('failure', per == 1, lambda pp: pp < 0.5), ('restart', per == 0, lambda pp: pp >= 0.5)):
                yy = df.y.values[cond]; pp = p[cond]; ev_true = (yy == 0) if ev == 'failure' else (yy == 1)
                pod = hit(pp)[ev_true].mean() if ev_true.any() else np.nan; far = hit(pp)[~ev_true].mean() if (~ev_true).any() else np.nan
                prec = ev_true[hit(pp)].mean() if hit(pp).any() else np.nan
                erow.append(dict(split=split, h=h, model=m, event=ev, n_events=int(ev_true.sum()), n_cases=int(cond.sum()), POD=pod, false_alarm_rate=far, precision=prec))
pd.DataFrame(rows).to_csv('res_brierdecomp.csv', index=False); pd.DataFrame(crow).to_csv('res_reliability.csv', index=False); pd.DataFrame(erow).to_csv('res_events.csv', index=False)
print(pd.DataFrame(rows).round(4).to_string()); print(pd.DataFrame(erow).round(3).to_string())
# lead time of failures: largest horizon at which the failure day was already predicted down (rolling design)
lt = []
P = {h: load('roll', h) for h in (1, 7, 14, 28)}
for m in ['Persistence', 'Physics prior', 'Data-driven ML', 'Proposed']:
    cnt = {0: 0, 1: 0, 7: 0, 14: 0, 28: 0}; tot = 0
    base = P[1][['date', 'node', 'y', 'Persistence']].rename(columns={'Persistence': 'prev'})
    fail = base[(base.prev == 1) & (base.y == 0)]  # first down day after an active day
    for _, r in fail.iterrows():
        tot += 1; best = 0
        for h in (1, 7, 14, 28):
            q = P[h]; row = q[(q.date == r.date) & (q.node == r.node)]
            if len(row) and row[m].values[0] < 0.5: best = h
        cnt[best] += 1
    lt.append(dict(model=m, n_failures=tot, **{f'lead_{k}': v for k, v in cnt.items()}))
pd.DataFrame(lt).to_csv('res_leadtime.csv', index=False); print(pd.DataFrame(lt).to_string())
