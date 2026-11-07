"""extended temporal experiments with additional baselines and ablations.
python3 exp3.py [tag] -> res3_temporal{tag}.csv and p3_temporal{tag}_<fold>_h<h>.csv
env PGL_THR (activity threshold, default 0.25)."""
import sys, os, numpy as np, pandas as pd, time, lightgbm as lgb, warnings
sys.path.insert(0, '.'); import core, phys2 as P, feats2 as F, models as M
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
warnings.filterwarnings('ignore')
TAG = sys.argv[1] if len(sys.argv) > 1 else ''
d = core.load(); days = d['days']; Tm = d['drv'].Tmean.values; Hc = d['drv'].Hclim.values
CAL = ['doy_sin', 'doy_cos', 'age']
def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def lr(tr, te, feats, C=1.0):
    if len(np.unique(tr.y)) < 2: return np.full(len(te), tr.y.mean()), np.full(len(tr), tr.y.mean())
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(tr[feats], tr.y)
    return m.predict_proba(te[feats])[:, 1], m.predict_proba(tr[feats])[:, 1]
def add_feats(D):
    D = D.copy(); D['lrun'] = np.log1p(D.run_len); D['act_x_lrun'] = D.act_t * D.lrun; D['mon_' + 'x'] = 0
    return D
def residual_on(tr, te, prior_tr, prior_te, feats, mono=None, seeds=5, n_est=150):
    out = np.zeros(len(te))
    for s in range(seeds):
        kw = dict(M.P, random_state=s, n_estimators=n_est)
        if mono: kw.update(monotone_constraints=[mono.get(f, 0) for f in feats], monotone_constraints_method='advanced')
        m = lgb.LGBMClassifier(**kw).fit(tr[feats], tr.y, init_score=logit(prior_tr))
        out += 1 / (1 + np.exp(-(m.predict(te[feats], raw_score=True) + logit(prior_te))))
    return out / seeds
GRID = [(n, l) for n in (100, 300, 600) for l in (7, 15, 31)]
def tuned(tr, te, feats, kind, mono=None):
    """nested time-series validation: the last 28 target days of the training period validate the grid"""
    last = tr.date.max(); vmask = tr.date > last - pd.Timedelta(days=28); trn, val = tr[~vmask], tr[vmask]
    best = None
    if len(np.unique(trn.y)) == 2 and len(val) > 50:
        for n, l in GRID:
            kw = dict(M.P, n_estimators=n, num_leaves=l, random_state=0)
            if mono: kw.update(monotone_constraints=[mono.get(f, 0) for f in feats], monotone_constraints_method='advanced')
            if kind == 'resid':
                ptr, pva = M.physics_prior(trn, val); m = lgb.LGBMClassifier(**kw).fit(trn[feats], trn.y, init_score=logit(ptr))
                p = 1 / (1 + np.exp(-(m.predict(val[feats], raw_score=True) + logit(pva))))
            else:
                m = lgb.LGBMClassifier(**kw).fit(trn[feats], trn.y); p = m.predict_proba(val[feats])[:, 1]
            b = np.mean((p - val.y.values) ** 2)
            if best is None or b < best[0]: best = (b, n, l)
    n, l = (best[1], best[2]) if best else (150 if kind == 'resid' else 300, 15)
    saveP = dict(M.P); M.P.update(num_leaves=l)
    try:
        p = M.residual_boost(tr, te, feats, mono=mono, n_est=n) if kind == 'resid' else M.gbm(tr, tr.y, te, feats, mono=mono) if n == 300 else _gbm_n(tr, te, feats, n, mono)
    finally:
        M.P.clear(); M.P.update(saveP)
    return p, (n, l)
def _gbm_n(tr, te, feats, n, mono=None):
    out = np.zeros(len(te))
    for s in range(5):
        kw = dict(M.P, random_state=s, n_estimators=n)
        if mono: kw.update(monotone_constraints=[mono.get(f, 0) for f in feats], monotone_constraints_method='advanced')
        out += lgb.LGBMClassifier(**kw).fit(tr[feats], tr.y).predict_proba(te[feats])[:, 1]
    return out / 5
def run_split(cut, t0_, t1_, H, tag):
    trmask = days <= cut; s_cal = np.array([P.calibrate(d, k, trmask, F.CFG, Hc, Tm) for k in range(20)]); res = []; tun = []
    for h in H:
        DF = add_feats(F.build(d, h, s_cal)); tr = DF[DF.date <= cut]; te = DF[(DF.date >= t0_) & (DF.date <= t1_)]
        if len(te) == 0: continue
        _, pphys = M.physics_prior(tr, te)
        seas, _ = lr(tr, te, ['doy_sin', 'doy_cos'])
        markov, _ = lr(tr, te, ['act_t', 'act_m7', 'act_m14', 'lrun', 'act_x_lrun', 'net_t', 'net_m7', 'doy_sin', 'doy_cos'])
        hazard, _ = lr(tr, te, ['act_t', 'lrun', 'act_x_lrun', 'doy_sin', 'doy_cos', 'age'])
        npte, nptr = lr(tr, te, ['act_t', 'lrun', 'act_x_lrun', 'doy_sin', 'doy_cos'])
        feats = F.DATA_F + F.PHYS_F; nocal = [f for f in feats if f not in CAL]
        p_tdd, cfg_dd = tuned(tr, te, F.DATA_F, 'gbm'); p_tpr, cfg_pr = tuned(tr, te, feats, 'resid', mono=F.MONO)
        preds = {'Persistence': te.act_t.values, 'Physics prior': pphys,
                 'Seasonal logistic (day of year)': seas, 'Logistic lag model (Markov-type)': markov, 'Discrete-time hazard model': hazard,
                 'ML data-driven': M.gbm(tr, tr.y, te, F.DATA_F), 'ML data-driven, tuned': p_tdd,
                 'Concatenation, constrained': M.gbm(tr, tr.y, te, feats, mono=F.MONO), 'Concatenation, unconstrained': M.gbm(tr, tr.y, te, feats),
                 'Residual on non-physical prior': residual_on(tr, te, nptr, npte, feats, mono=F.MONO),
                 'Proposed': M.residual_boost(tr, te, feats, mono=F.MONO), 'Proposed, tuned': p_tpr,
                 'Proposed without calendar features': M.residual_boost(tr, te, nocal, mono=F.MONO)}
        for name, p in preds.items():
            r = M.evaluate(te, np.asarray(p, float)); r.update(model=name, h=h, split=tag); res.append(r)
        tun.append(dict(split=tag, h=h, dd_n=cfg_dd[0], dd_leaves=cfg_dd[1], pr_n=cfg_pr[0], pr_leaves=cfg_pr[1]))
        pd.DataFrame(dict(date=te.date.values, node=te.node.values, y=te.y.values, act_t=te.act_t.values, **{n: np.asarray(p, float) for n, p in preds.items()})).to_csv(f'p3_temporal{TAG}_{tag}_h{h}.csv', index=False)
        print(tag, h, len(tr), len(te), round(time.time() - T0), flush=True)
    return res, tun
T0 = time.time(); R = []; TU = []
r, t = run_split(pd.Timestamp('2025-12-31'), pd.Timestamp('2026-01-01'), pd.Timestamp('2026-06-13'), (7, 14, 28), 'fixed'); R += r; TU += t
for M0 in pd.date_range('2025-11-01', '2026-06-01', freq='MS'):
    r, t = run_split(M0 - pd.Timedelta(days=1), M0, M0 + pd.offsets.MonthEnd(0), (7, 14, 28), 'roll_' + M0.strftime('%Y-%m')); R += r; TU += t
pd.DataFrame(R).to_csv(f'res3_temporal{TAG}.csv', index=False); pd.DataFrame(TU).to_csv(f'res3_tuning{TAG}.csv', index=False); print('done', time.time() - T0)
