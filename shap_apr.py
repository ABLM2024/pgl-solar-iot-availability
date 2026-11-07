"""feature attributions (TreeSHAP, pred_contrib) of the proposed learner for April 2026 (h=28) and December 2025 (h=14);
and diagnosis of the December failure. -> res_shap.csv, res_dec_diag.csv"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '.'); import core, phys2 as P, feats2 as F, models as M
d = core.load(); days = d['days']; Tm = d['drv'].Tmean.values; Hc = d['drv'].Hclim.values
CAL = ['doy_sin', 'doy_cos', 'age']
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
def fam(f):
    if f in CAL: return 'calendar'
    if f in F.PHYS_F: return 'physics features'
    return 'data features'
rows = []; diag = []
for M0, h in ((pd.Timestamp('2026-04-01'), 28), (pd.Timestamp('2025-12-01'), 14), (pd.Timestamp('2025-12-01'), 28), (pd.Timestamp('2026-03-01'), 28)):
    cut = M0 - pd.Timedelta(days=1); trmask = days <= cut
    s_cal = np.array([P.calibrate(d, k, trmask, F.CFG, Hc, Tm) for k in range(20)])
    DF = F.build(d, h, s_cal); tr = DF[DF.date <= cut]; te = DF[(DF.date >= M0) & (DF.date <= M0 + pd.offsets.MonthEnd(0))]
    feats = F.DATA_F + F.PHYS_F; ptr, pte = M.physics_prior(tr, te)
    con = np.zeros((len(te), len(feats) + 1))
    for s in range(5):
        kw = dict(M.P, random_state=s, n_estimators=150, monotone_constraints=[F.MONO.get(f, 0) for f in feats], monotone_constraints_method='advanced')
        m = lgb.LGBMClassifier(**kw).fit(tr[feats], tr.y, init_score=lg(ptr))
        con += m.booster_.predict(te[feats], pred_contrib=True) / 5
    contrib = pd.DataFrame(con[:, :-1], columns=feats)
    tot_logit = lg(pte) + con.sum(1); p = 1 / (1 + np.exp(-tot_logit))
    lab = f'{M0:%Y-%m} h={h}'
    # mean contribution (logit) and mean absolute contribution per family; prior logit
    for fm in ('calendar', 'physics features', 'data features'):
        cols = [f for f in feats if fam(f) == fm]
        rows.append(dict(fold=lab, family=fm, mean_contrib=contrib[cols].sum(1).mean(), mean_abs_contrib=contrib[cols].sum(1).abs().mean()))
    rows.append(dict(fold=lab, family='physics prior (init score)', mean_contrib=lg(pte).mean(), mean_abs_contrib=np.abs(lg(pte)).mean()))
    top = contrib.abs().mean().sort_values(ascending=False).head(8)
    for f, v in top.items(): rows.append(dict(fold=lab, family='top: ' + f, mean_contrib=contrib[f].mean(), mean_abs_contrib=v))
    # December diagnosis: state at origin and predicted vs observed network availability
    diag.append(dict(fold=lab, n_train=len(tr), train_last_target=str(tr.date.max().date()), train_pos=tr.y.mean(), test_pos=te.y.mean(),
                     act_at_origin=te.act_t.mean(), prior_mean=pte.mean(), proposed_mean=p.mean(), up_sim_mean=te.up_sim.mean(), latched_mean=te.latched_t.mean(),
                     soc_h_mean=te.soc_h.mean(), train_winter_downfrac=(1 - tr[tr.date >= cut - pd.Timedelta(days=30)].y.mean())))
R = pd.DataFrame(rows); R.to_csv('res_shap.csv', index=False); Dg = pd.DataFrame(diag); Dg.to_csv('res_dec_diag.csv', index=False)
pd.set_option('display.width', 250); print(R.round(3).to_string()); print(Dg.round(3).to_string())
