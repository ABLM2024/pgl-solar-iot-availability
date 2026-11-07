"""Holm-adjusted weekly block bootstrap: family 1 = 48 pre-defined comparisons (res_boot_temporal.csv, recomputed here),
family 2 = proposed vs the additional baselines. -> res_boot_all.csv"""
import glob, numpy as np, pandas as pd
NAMES = {'Hybrid: physics-prior residual (proposed)': 'Proposed', 'ML data-driven': 'Data-driven ML', 'Physics v2 assimilated (calibrated prob.)': 'Physics prior'}
def load(pat, split, h):
    fs = sorted(glob.glob(f'{pat}_{split}*_h{h}.csv')); return pd.concat([pd.read_csv(f, parse_dates=['date']) for f in fs]).rename(columns=NAMES)
def boot(df, a, b, metric, B=4000, seed=0):
    df = df.copy()
    if metric == 'brier':
        df['ea'] = (df[a] - df.y) ** 2; df['eb'] = (df[b] - df.y) ** 2; unit = df
    else:
        unit = df.groupby('date').agg(y=('y', 'mean'), pa=(a, 'mean'), pb=(b, 'mean')).reset_index()
        unit['ea'] = 100 * np.abs(unit.pa - unit.y); unit['eb'] = 100 * np.abs(unit.pb - unit.y)
    unit['wk'] = unit.date.dt.to_period('W').astype(str)
    g = unit.groupby('wk')[['ea', 'eb']].agg(['sum', 'count']); sa = g[('ea', 'sum')].values; sb = g[('eb', 'sum')].values; nn = g[('ea', 'count')].values
    est = (sa.sum() - sb.sum()) / nn.sum(); rng = np.random.default_rng(seed); W = len(nn); bs = np.empty(B)
    for i in range(B):
        j = rng.integers(0, W, W); bs[i] = (sa[j].sum() - sb[j].sum()) / nn[j].sum()
    lo, hi = np.percentile(bs, [2.5, 97.5]); p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return est, lo, hi, max(p, 1 / B), W
def holm(p):
    o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj
rows = []
for split in ('fixed', 'roll'):
    for h in (1, 7, 14, 28):
        df = load('p2_temporal', split, h)
        for b in ('Physics prior', 'Persistence', 'Data-driven ML'):
            for met in ('brier', 'net'):
                e, lo, hi, p, W = boot(df, 'Proposed', b, met); rows.append(dict(family=1, split=split, h=h, vs=b, metric=met, diff=e, lo=lo, hi=hi, p=p, weeks=W))
        if h == 1: continue
        df = load('p3_temporal', split, h)
        for b in ('ML data-driven, tuned', 'Discrete-time hazard model', 'Logistic lag model (Markov-type)', 'Seasonal logistic (day of year)', 'Residual on non-physical prior'):
            for met in ('brier', 'net'):
                e, lo, hi, p, W = boot(df, 'Proposed', b, met); rows.append(dict(family=2, split=split, h=h, vs=b, metric=met, diff=e, lo=lo, hi=hi, p=p, weeks=W))
        # the best statistical baseline against the data-driven learner (context)
        for met in ('brier', 'net'):
            e, lo, hi, p, W = boot(df, 'Discrete-time hazard model', 'Data-driven ML', met); rows.append(dict(family=3, split=split, h=h, vs='hazard minus data-driven', metric=met, diff=e, lo=lo, hi=hi, p=p, weeks=W))
R = pd.DataFrame(rows); R['p_holm'] = np.nan
for f in (1, 2, 3):
    m = R.family == f; R.loc[m, 'p_holm'] = holm(R.loc[m, 'p'].values)
R.to_csv('res_boot_all.csv', index=False); pd.set_option('display.width', 250); print(R.round(4).to_string())
