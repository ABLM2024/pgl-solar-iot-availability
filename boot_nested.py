"""Weekly block bootstrap with Holm adjustment for the proposed learner with the
physics structure selected inside each rolling fold (exp_nested.py) against
persistence, the data-driven learner, the discrete-time hazard model and the
proposed learner with the original structure (exp3.py).
python3 boot_nested.py [prediction_dir] -> res_boot_nested.csv"""
import sys, glob, numpy as np, pandas as pd
D = sys.argv[1] if len(sys.argv) > 1 else '.'
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
def cat(pat):
    return pd.concat([pd.read_csv(f, parse_dates=['date']) for f in sorted(glob.glob(pat))])
rows = []
VS = ['Persistence', 'ML data-driven', 'Discrete-time hazard model', 'Proposed']
for h in (7, 14, 28):
    n = cat(f'{D}/p_nested_roll_*_h{h}.csv')[['date', 'node', 'y', 'proposed']].rename(columns={'proposed': 'nested'})
    b = cat(f'{D}/p3_temporal_roll_*_h{h}.csv')[['date', 'node', 'y'] + VS].rename(columns={'y': 'y3'})
    df = n.merge(b, on=['date', 'node'], how='inner'); assert (df.y == df.y3).all() and len(df) == len(n)
    for v in VS:
        for met in ('brier', 'net'):
            e, lo, hi, p, W = boot(df, 'nested', v, met); rows.append(dict(h=h, vs=v, metric=met, diff=e, lo=lo, hi=hi, p=p, weeks=W, n=len(df)))
R = pd.DataFrame(rows); R['p_holm'] = holm(R.p.values)
R.to_csv('res_boot_nested.csv', index=False); pd.set_option('display.width', 200); print(R.round(4).to_string())
