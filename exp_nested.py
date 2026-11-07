"""physics-structure selection nested inside every fold, on an extended grid; fold-specific structure used for the prior/proposed.
-> res_nested.csv, res_structgrid_fixed.csv, p_nested_<fold>_h<h>.csv"""
import sys, numpy as np, pandas as pd, itertools, time
sys.path.insert(0, '.'); import core, phys2 as P, feats2 as F, models as M
d = core.load(); days = d['days']; Tm = d['drv'].Tmean.values; Hc = d['drv'].Hclim.values; n = len(days)
GRID = list(itertools.product([0.5, 0.7, 0.9, 0.95, 1.0], [0.3, 0.5, 0.8], [1, 2, 3, 7, 14, 999], [5.0]))
GRID += [(0.9, 0.5, 3, tl) for tl in (0.0, 2.5, 7.5, 10.0)]
def score(cfg, trmask, evalmask=None):
    bal = []; bal_te = []
    for k in range(20):
        s = P.calibrate(d, k, trmask, cfg, Hc, Tm); st = int(np.argmax(d['valid'][:, k])); _, up, _ = P.run(Hc, Tm, s, cfg, st, n)
        for msk, out in ((trmask, bal), (evalmask, bal_te)):
            if msk is None: continue
            m = msk & d['valid'][:, k] & ~d['cm']; y = d['act'][m, k]; p = up[m]
            if len(y) == 0: continue
            out.append(np.nanmean([p[y == 1].mean() if (y == 1).any() else np.nan, (~p[y == 0]).mean() if (y == 0).any() else np.nan]))
    return np.nanmean(bal), (np.nanmean(bal_te) if bal_te else np.nan)
T0 = time.time(); rows = []; G = []
folds = [('fixed', pd.Timestamp('2025-12-31'), pd.Timestamp('2026-01-01'), pd.Timestamp('2026-06-13'))] + \
        [('roll_' + M0.strftime('%Y-%m'), M0 - pd.Timedelta(days=1), M0, M0 + pd.offsets.MonthEnd(0)) for M0 in pd.date_range('2025-11-01', '2026-06-01', freq='MS')]
for tag, cut, t0_, t1_ in folds:
    trm = days <= cut; tem = (days >= t0_) & (days <= t1_)
    sc = []
    for g in GRID:
        cfg = P.Cfg(dod=g[0], R=g[1], latch_days=g[2], T_latch=g[3], driver='Hclim'); b, bt = score(cfg, trm, tem if tag == 'fixed' else None)
        sc.append((b, g)); 
        if tag == 'fixed': G.append(dict(dod=g[0], R=g[1], N_L=g[2], T_L=g[3], bacc_train=b, bacc_test_openloop=bt))
    b, g = max(sc, key=lambda x: x[0]); cfg = P.Cfg(dod=g[0], R=g[1], latch_days=g[2], T_latch=g[3], driver='Hclim')
    F.CFG = cfg; s_cal = np.array([P.calibrate(d, k, trm, cfg, Hc, Tm) for k in range(20)])
    for h in (7, 14, 28):
        DF = F.build(d, h, s_cal); tr = DF[DF.date <= cut]; te = DF[(DF.date >= t0_) & (DF.date <= t1_)]
        _, pphys = M.physics_prior(tr, te); pp = M.residual_boost(tr, te, F.DATA_F + F.PHYS_F, mono=F.MONO)
        for name, p in (('Physics prior (nested structure)', pphys), ('Proposed (nested structure)', pp)):
            r = M.evaluate(te, p); r.update(model=name, h=h, split=tag, dod=g[0], R=g[1], N_L=g[2], T_L=g[3], bacc_sel=b); rows.append(r)
        pd.DataFrame(dict(date=te.date.values, node=te.node.values, y=te.y.values, prior=pphys, proposed=pp)).to_csv(f'p_nested_{tag}_h{h}.csv', index=False)
    print(tag, g, round(b, 3), round(time.time() - T0), flush=True)
    if tag == 'fixed': pd.DataFrame(G).sort_values('bacc_train', ascending=False).to_csv('res_structgrid_fixed.csv', index=False)
pd.DataFrame(rows).to_csv('res_nested.csv', index=False); print('done')
