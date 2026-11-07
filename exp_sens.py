"""sensitivity of physics and forecasts to fixed parameters (env vars PGL_TL, PGL_RES, PGL_PR, PGL_NL, PGL_DOD) and the common-mode rule.
python3 exp_sens.py LABEL -> appends to res_sens.csv"""
import sys, os, numpy as np, pandas as pd, time
sys.path.insert(0, '.'); import core, phys2 as P, feats2 as F, models as M
LAB = sys.argv[1]; d = core.load(); days = d['days']; Tm = d['drv'].Tmean.values; Hc = d['drv'].Hclim.values; n = len(days)
if os.environ.get('PGL_CM') == 'off': d['cm'][:] = False
if os.environ.get('PGL_CM') == 'dec': d['cm'][(days >= pd.Timestamp('2025-12-06')) & (days <= pd.Timestamp('2025-12-08'))] = True
rows = []
trm = days <= pd.Timestamp('2025-12-31'); tem = ~trm
bal = []; balt = []
for k in range(20):
    s = P.calibrate(d, k, trm, F.CFG, Hc, Tm); st = int(np.argmax(d['valid'][:, k])); _, up, _ = P.run(Hc, Tm, s, F.CFG, st, n)
    for msk, out in ((trm, bal), (tem, balt)):
        m = msk & d['valid'][:, k] & ~d['cm']; y = d['act'][m, k]; p = up[m]
        out.append(np.nanmean([p[y == 1].mean() if (y == 1).any() else np.nan, (~p[y == 0]).mean() if (y == 0).any() else np.nan]))
base = dict(label=LAB, bacc_train=np.nanmean(bal), bacc_test_openloop=np.nanmean(balt))
folds = [('fixed', pd.Timestamp('2025-12-31'), pd.Timestamp('2026-01-01'), pd.Timestamp('2026-06-13'))] + \
        [('roll_' + M0.strftime('%Y-%m'), M0 - pd.Timedelta(days=1), M0, M0 + pd.offsets.MonthEnd(0)) for M0 in pd.date_range('2025-11-01', '2026-06-01', freq='MS')]
if os.environ.get('PGL_FOLDS', 'all') == 'fixed': folds = folds[:1]
for tag, cut, t0_, t1_ in folds:
    trm_ = days <= cut; s_cal = np.array([P.calibrate(d, k, trm_, F.CFG, Hc, Tm) for k in range(20)])
    DF = F.build(d, 28, s_cal); tr = DF[DF.date <= cut]; te = DF[(DF.date >= t0_) & (DF.date <= t1_)]
    _, pphys = M.physics_prior(tr, te); pp = M.residual_boost(tr, te, F.DATA_F + F.PHYS_F, mono=F.MONO)
    pd.DataFrame(dict(date=te.date.values, node=te.node.values, y=te.y.values, prior=pphys, proposed=pp)).to_csv(f'p_sens_{LAB}_{tag}_h28.csv', index=False)
    for name, p in (('Physics prior', pphys), ('Proposed', pp)):
        r = M.evaluate(te, p); r.update(base, model=name, split=tag); rows.append(r)
R = pd.DataFrame(rows); f = 'res_sens.csv'
R.to_csv(f, mode='a', header=not os.path.exists(f), index=False); print(LAB, base, R[R.split != 'fixed'].groupby('model')[['brier', 'net_mae_pp']].mean().round(3).to_dict())
