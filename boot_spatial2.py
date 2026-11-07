"""held-out-node differences with weekly, node-cluster and two-way (node x week) bootstrap; Holm over the 36 comparisons.
-> boot_spatial.csv (weekly) and res_boot_spatial.csv (all schemes)"""
import numpy as np, pandas as pd
FILES = {'own': 'p2_spatial_Spatio-temporal_own_calibration_h{h}.csv', 'median': 'p2_spatial_LONO-spatiotemporal_h{h}.csv'}
PROP = 'Hybrid: physics-prior residual (proposed)'; VS = ['ML data-driven', 'Persistence', 'Physics v2 calibrated prob.']
B = 2000
def mats(df, a, b):
    df = df.copy(); df['date'] = pd.to_datetime(df.date)
    nodes = np.sort(df.node.unique()); dates = np.sort(df.date.unique())
    ix = {n: i for i, n in enumerate(nodes)}; jx = {t: j for j, t in enumerate(dates)}
    Y = np.full((len(nodes), len(dates)), np.nan); PA = Y.copy(); PB = Y.copy()
    for n, t, y, pa, pb in zip(df.node, df.date, df.y, df[a], df[b]):
        Y[ix[n], jx[t]] = y; PA[ix[n], jx[t]] = pa; PB[ix[n], jx[t]] = pb
    wk = pd.Series(dates).dt.to_period('W').astype(str).values
    return Y, PA, PB, wk
def stat(Y, PA, PB, wn, wd, metric):
    M = ~np.isnan(Y)
    if metric == 'brier':
        W = wn[:, None] * wd[None, :] * M
        ea = np.nan_to_num((PA - Y) ** 2); eb = np.nan_to_num((PB - Y) ** 2)
        return ((ea - eb) * W).sum() / W.sum()
    W = wn[:, None] * M; den = W.sum(0); ok = den > 0
    ya = (np.nan_to_num(Y) * W).sum(0)[ok] / den[ok]; pa = (np.nan_to_num(PA) * W).sum(0)[ok] / den[ok]; pb = (np.nan_to_num(PB) * W).sum(0)[ok] / den[ok]
    d = 100 * (np.abs(pa - ya) - np.abs(pb - ya)); v = wd[ok]
    return (d * v).sum() / v.sum()
rows = []
for setting, pat in FILES.items():
    for h in (1, 7, 28):
        df = pd.read_csv(pat.format(h=h))
        for b in VS:
            Y, PA, PB, wk = mats(df, PROP, b); nN = Y.shape[0]; uw = np.unique(wk); wid = np.array([np.where(uw == w)[0][0] for w in wk])
            for metric in ('brier', 'net'):
                est = stat(Y, PA, PB, np.ones(nN), np.ones(len(wk)), metric)
                rng = np.random.default_rng(0); res = {}
                for scheme in ('week', 'node', 'two-way'):
                    bs = np.empty(B)
                    for i in range(B):
                        wn = np.bincount(rng.integers(0, nN, nN), minlength=nN).astype(float) if scheme != 'week' else np.ones(nN)
                        wd = np.bincount(rng.integers(0, len(uw), len(uw)), minlength=len(uw)).astype(float)[wid] if scheme != 'node' else np.ones(len(wk))
                        bs[i] = stat(Y, PA, PB, wn, wd, metric)
                    lo, hi = np.percentile(bs, [2.5, 97.5]); p = max(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())), 1 / B)
                    res[scheme] = (lo, hi, p)
                rows.append(dict(setting=setting, h=h, vs=b, metric=metric, diff=est, **{f'{s}_{k}': v for s in res for k, v in zip(('lo', 'hi', 'p'), res[s])}))
        print(setting, h, flush=True)
R = pd.DataFrame(rows)
def holm(p):
    o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj
for s in ('week', 'node', 'two-way'): R[f'{s}_p_holm'] = holm(R[f'{s}_p'].values)
R.to_csv('res_boot_spatial.csv', index=False)
R[['setting', 'h', 'vs', 'metric', 'diff', 'week_lo', 'week_hi']].rename(columns={'week_lo': 'lo', 'week_hi': 'hi'}).to_csv('boot_spatial.csv', index=False)
pd.set_option('display.width', 250); print(R.round(4).to_string())
