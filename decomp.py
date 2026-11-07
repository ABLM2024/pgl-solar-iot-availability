"""as-installed physics, decomposition of down node-days and their bias contributions,
panel-size counterfactual inside the calibrated model with uncertainty. -> res_decomp.csv, res_panel.csv, res_panel_node.csv"""
import sys, numpy as np, pandas as pd, itertools
sys.path.insert(0, '.'); import core, phys2 as P, feats2 as F
d = core.load(); days = d['days']; D = core.D; L = core.L
ref = pd.read_csv(D + 'C_reference/reference_PM25_daily_EEA.csv', parse_dates=['date']).set_index('date').reindex(days)
SITES = {'A': 'PM25_siteA_SPO_RO0065A', 'B': 'PM25_siteB_SPO_RO0229A'}
Hc = d['drv'].Hclim.values; Tm = d['drv'].Tmean.values; VAL = d['valid']; ACT = d['act'] * VAL; CM = d['cm']; n = len(days)
allm = np.ones(n, bool)
def beta(a, lim=15.0):
    out = {}
    for s, col in SITES.items():
        c = ref[col].values; m = ~np.isnan(c); aa = a[m]; cc = c[m]; tru = cc.mean(); exc = cc > lim
        out[f'bias_{s}'] = 100 * ((aa * cc).sum() / aa.sum() - tru) / tru; out[f'exc_{s}'] = 100 * aa[exc].sum() / exc.sum()
    out['mean_avail'] = 100 * a.mean(); return out
def ba_profile(k, cfg, grid=np.round(np.arange(0.3, 3.01, 0.1), 2), mask=allm, H=Hc):
    st = int(np.argmax(VAL[:, k])); m = mask & VAL[:, k] & ~CM; y = d['act'][m, k]; out = []
    for s in grid:
        _, up, _ = P.run(H, Tm, s, cfg, st, n); p = up[m]
        tpr = p[y == 1].mean() if (y == 1).any() else np.nan; tnr = (~p[y == 0]).mean() if (y == 0).any() else np.nan
        out.append((s, np.nanmean([tpr, tnr])))
    return np.array(out)
CFG = F.CFG
# multipliers calibrated on the full year (as in the counterfactuals) and on the training half, with balanced-accuracy profiles
prof = {k: ba_profile(k, CFG) for k in range(20)}
prof_tr = {k: ba_profile(k, CFG, mask=days <= pd.Timestamp('2025-12-31')) for k in range(20)}
def best(pr):
    mx = np.nanmax(pr[:, 1]); cand = pr[pr[:, 1] >= mx - 1e-12, 0]; return cand[np.argmin(np.abs(cand - 1))], mx
S = np.array([best(prof[k])[0] for k in range(20)])
rows_s = []
for k in range(20):
    for nm, pr in (('full', prof[k]), ('train', prof_tr[k])):
        s0, mx = best(pr); near = pr[pr[:, 1] >= mx - 0.01, 0]
        rows_s.append(dict(node=int(d['nodes'][k]), mount=d['mount'][k], period=nm, s_best=s0, ba_max=mx, near_lo=near.min(), near_hi=near.max(),
                           censored_upper=bool(near.max() >= 2.95), censored_lower=bool(near.min() <= 0.35)))
pd.DataFrame(rows_s).to_csv('res_s_profile.csv', index=False)
pd.DataFrame({f'node{int(d["nodes"][k])}': prof_tr[k][:, 1] for k in range(20)}, index=prof_tr[0][:, 0]).to_csv('res_s_profile_curves.csv')
def sim(Svec, cfg=CFG, fac=1.0, H=Hc, Tmx=Tm):
    U = np.zeros((n, 20), bool); NET = np.zeros((n, 20))
    for k in range(20):
        st = int(np.argmax(VAL[:, k])); U[:, k] = P.run(H, Tmx, Svec[k] * fac, cfg, st, n)[1]
        NET[:, k] = core.PANEL * Svec[k] * fac * H * L.PR - 24
    return U & VAL, NET
U, NET = sim(S)
rows = [dict(source='Measured (active node-days / 20)', **beta(ACT.sum(1) / 20)),
        dict(source='Physics, as installed (10 Wp, open-loop, full-year calibration)', **beta(U.sum(1) / 20)),
        dict(source='All commissioned nodes up', **beta(VAL.sum(1) / 20))]
# decomposition of measured down node-days (commissioned, not common-mode)
down = VAL & (ACT == 0) & ~CM[:, None]
energy = down & (NET < 0)
# spells ending with the collective restarts of 18 March and 6 May 2026
maint = np.zeros_like(down)
for k in range(20):
    t = 0
    while t < n:
        if down[t, k]:
            j = t
            while j + 1 < n and (down[j + 1, k] or CM[j + 1]): j += 1
            end_next = days[j + 1] if j + 1 < n else None
            if end_next is not None and end_next in (pd.Timestamp('2026-03-18'), pd.Timestamp('2026-05-06')):
                maint[t:j + 1, k] = down[t:j + 1, k] & (NET[t:j + 1, k] >= 0)
            t = j + 1
        else: t += 1
other = down & ~energy & ~maint
tot = down.sum()
dec = dict(down=int(tot), energy=int(energy.sum()), maint=int(maint.sum()), other=int(other.sum()), cm_days=int((VAL & CM[:, None]).sum()))
# where are the 'other' days
oth_nodes = {int(d['nodes'][k]): int(other[:, k].sum()) for k in range(20) if other[:, k].sum() > 0}
dec['other_by_node'] = str(oth_nodes)
# bias contributions: restore one class at a time, and in sequence
base = ACT.astype(float).copy()
for lab, mask in (('restore maintenance-class down days', maint), ('restore other non-energy down days', other), ('restore common-mode days', VAL & CM[:, None]),
                  ('restore energy-deficit down days', energy)):
    a = np.where(mask, 1.0, base); rows.append(dict(source='Measured + ' + lab, **beta(a.sum(1) / 20)))
a = np.where(maint | other | (VAL & CM[:, None]), 1.0, base); rows.append(dict(source='Measured + all non-energy down days restored', **beta(a.sum(1) / 20)))
# panel-size counterfactuals within the calibrated model
DJF = np.isin(days.month.values, [12, 1, 2])
def djf_uptime(Uc): v = VAL & DJF[:, None]; return 100 * (Uc & v).sum() / v.sum()
prow = []
for wp in (10, 15, 16, 20, 21, 25, 30, 40, 50, 60):
    Uc, _ = sim(S, fac=wp / 10)
    r = dict(Wp=wp, djf_uptime=djf_uptime(Uc), **beta(Uc.sum(1) / 20))
    Uc2, _ = sim(S, cfg=P.Cfg(dod=CFG.dod, R=CFG.R, latch_days=999, T_latch=CFG.T_latch, driver=CFG.driver), fac=wp / 10)
    r.update(djf_uptime_nolatch=djf_uptime(Uc2), **{k + '_nolatch': v for k, v in beta(Uc2.sum(1) / 20).items()})
    prow.append(r); print(r, flush=True)
    if wp == 21: rows.append(dict(source='Physics counterfactual, 21 Wp', djf_uptime=r['djf_uptime'], **beta(Uc.sum(1) / 20)))
# per-node panel size for 99 % DJF uptime (with latch and without)
nrow = []
for k in range(20):
    st = int(np.argmax(VAL[:, k])); v = VAL[:, k] & DJF
    if v.sum() == 0: continue
    rk = dict(node=int(d['nodes'][k]), s=S[k])
    for lab, cfg in (('latch', CFG), ('nolatch', P.Cfg(dod=CFG.dod, R=CFG.R, latch_days=999, T_latch=CFG.T_latch, driver=CFG.driver))):
        need = np.nan
        for wp in np.arange(10, 101, 1):
            up = P.run(Hc, Tm, S[k] * wp / 10, cfg, st, n)[1]
            if (up & v).sum() / v.sum() >= 0.99: need = wp; break
        rk['Wp99_' + lab] = need
    nrow.append(rk)
N = pd.DataFrame(nrow); N.to_csv('res_panel_node.csv', index=False); print(N.to_string())
# uncertainty of the 21-Wp counterfactual: near-optimal multipliers, top structures, winter irradiation +-15 % (assumed)
SS = pd.read_csv('structure_selection.csv'); top = SS[SS.driver == 'Hclim'].head(5)
urow = []
for lab, Svec in (('s low (near-optimal)', np.array([r['near_lo'] for r in rows_s if r['period'] == 'full'])), ('s high (near-optimal)', np.array([r['near_hi'] for r in rows_s if r['period'] == 'full']))):
    Uc, _ = sim(Svec, fac=2.1); urow.append(dict(case=lab, djf_uptime=djf_uptime(Uc), **beta(Uc.sum(1) / 20)))
for _, r in top.iterrows():
    cfg = P.Cfg(dod=r.dod, R=r.R, latch_days=int(r.latch), T_latch=5.0, driver='Hclim')
    Sv = np.array([best(ba_profile(k, cfg))[0] for k in range(20)])
    Uc, _ = sim(Sv, cfg=cfg, fac=2.1); U0, _ = sim(Sv, cfg=cfg)
    urow.append(dict(case=f'structure dod={r.dod}, R={r.R}, N_L={int(r.latch)}', djf_uptime=djf_uptime(Uc), djf_uptime_10Wp=djf_uptime(U0), **beta(Uc.sum(1) / 20)))
for fw in (0.85, 1.15):
    Hs = Hc * np.where(np.isin(days.month.values, [11, 12, 1, 2]), fw, 1.0)
    Uc, _ = sim(S, fac=2.1, H=Hs); urow.append(dict(case=f'winter irradiation x{fw}', djf_uptime=djf_uptime(Uc), **beta(Uc.sum(1) / 20)))
pd.DataFrame(urow).to_csv('res_panel_unc.csv', index=False); print(pd.DataFrame(urow).round(1).to_string())
R = pd.DataFrame(rows); R.to_csv('res_decomp.csv', index=False); pd.DataFrame(prow).to_csv('res_panel.csv', index=False)
pd.DataFrame([dec]).to_csv('res_decomp_counts.csv', index=False)
print(R.round(1).to_string()); print(dec)
