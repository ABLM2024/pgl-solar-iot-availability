"""target statistics and estimators of the network PM2.5 statistic under the reference-proxy assumption.
Outputs res_estimators.csv, res_nodecapture.csv, res_proxy_bounds.csv, res_deposit_series.csv"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.'); import core
d = core.load(); days = d['days']; D = core.D
ref = pd.read_csv(D + 'C_reference/reference_PM25_daily_EEA.csv', parse_dates=['date']).set_index('date').reindex(days)
SITES = {'A': 'PM25_siteA_SPO_RO0065A', 'B': 'PM25_siteB_SPO_RO0229A'}
FR = d['A']; VAL = d['valid']
def active(thr): return ((FR >= thr) & VAL).astype(float)
def week_ids(): return pd.Series(days).dt.to_period('W').astype(str).values
WK = week_ids()

def est_all(c, act, lim=15.0, val=None):
    val = VAL if val is None else val
    """returns dict of statistics for the three estimators; c daily ref conc (NaN allowed)"""
    m = ~np.isnan(c); cc = c[m]; a = act[m]; n_on = a.sum(1); true = cc.mean(); exc = cc > lim
    out = dict(true=true, n_days=int(m.sum()), exc_days=int(exc.sum()))
    # (i) pooled node-day mean (Eq. 15 basis: A_t = active nodes / 20)
    A = n_on / 20.0; out['pooled_bias'] = 100 * ((A * cc).sum() / A.sum() - true) / true
    out['pooled_exc'] = 100 * A[exc].sum() / max(exc.sum(), 1)
    # (ii) daily network mean, day counted if >= k nodes report
    for k in (1, 2, 5):
        on = n_on >= k
        out[f'daily{k}_days'] = int(on.sum()); out[f'daily{k}_bias'] = 100 * (cc[on].mean() - true) / true
        out[f'daily{k}_exc'] = 100 * (on & exc).sum() / max(exc.sum(), 1)
    # (iii) per-node annual means over the commissioned period of each node
    pb = []; pe = []; cap = []
    for k in range(a.shape[1]):
        v = val[m, k]; on = a[:, k] == 1
        if v.sum() == 0: continue
        tk = cc[v].mean(); cap.append(100 * on.sum() / v.sum())
        pb.append(100 * (cc[on].mean() - tk) / tk if on.any() else np.nan)
        ex = exc & v; pe.append(100 * (on & ex).sum() / max(ex.sum(), 1))
    pb = np.array(pb); pe = np.array(pe); cap = np.array(cap)
    out.update(node_bias_median=np.nanmedian(pb), node_bias_min=np.nanmin(pb), node_bias_max=np.nanmax(pb),
               node_exc_median=np.median(pe), n_nodes_75=int((cap >= 75).sum()), n_nodes_90=int((cap >= 90).sum()))
    return out, pb, pe, cap

def boot(c, act, key, B=2000, seed=0, lim=15.0):
    rng = np.random.default_rng(seed); wks = np.unique(WK); idx = {w: np.where(WK == w)[0] for w in wks}; vals = []
    for _ in range(B):
        sel = np.concatenate([idx[w] for w in rng.choice(wks, len(wks))])
        vals.append(est_all(c[sel], act[sel], lim, VAL[sel])[0][key])
    return np.nanpercentile(vals, [2.5, 97.5])

rows = []; caprows = []
for thr in (0.25, 0.75):
    act = active(thr)
    for lim in (15.0, 25.0):
        for s, col in SITES.items():
            c = ref[col].values; o, pb, pe, cap = est_all(c, act, lim)
            o.update(site=s, thr=thr, limit=lim, active_frac=100 * act.sum() / VAL.sum())
            if lim == 15.0:
                for key in ('pooled_bias', 'daily1_bias', 'daily2_bias', 'node_bias_median'):
                    lo, hi = boot(c, act, key, B=1000); o[key + '_lo'] = lo; o[key + '_hi'] = hi
            rows.append(o)
            if lim == 15.0 and thr == 0.25:
                for k in range(20):
                    pass
    if thr == 0.25:
        m = ~np.isnan(ref[SITES['A']].values)
        for k in range(20):
            v = VAL[:, k]; on = (act[:, k] == 1)
            r = dict(node=int(d['nodes'][k]), mount=d['mount'][k], commissioned=str(pd.Timestamp(d['com'][k]).date()), capture_pct=100 * on[v].sum() / v.sum(),
                     capture_cal_pct=100 * on.sum() / len(days))
            for s, col in SITES.items():
                c = ref[col].values; mm = ~np.isnan(c) & v; tk = c[mm].mean(); oo = mm & on
                r[f'bias_{s}'] = 100 * (c[oo].mean() - tk) / tk if oo.any() else np.nan
                ex = mm & (c > 15); r[f'exc_{s}'] = 100 * (ex & on).sum() / max(ex.sum(), 1)
            r['meets75'] = r['capture_pct'] >= 75; r['meets90'] = r['capture_pct'] >= 90; caprows.append(r)
R = pd.DataFrame(rows); R.to_csv('res_estimators.csv', index=False)
C = pd.DataFrame(caprows); C.to_csv('res_nodecapture.csv', index=False)
pd.set_option('display.width', 250); print(R.round(1).to_string()); print(C.round(1).to_string())

# deposit series vs active node-days / 20
dep = ref.network_availability_fraction.values; rec = (active(0.25)).sum(1) / 20; mfrac = np.where(VAL, FR, 0).sum(1) / 20
X = pd.DataFrame(dict(date=days, deposit=dep, active_over_20=rec, mean_fraction_over_20=mfrac))
X.to_csv('res_deposit_series.csv', index=False)
print('deposit vs mean fraction: max abs diff', np.nanmax(np.abs(dep - mfrac)), ' vs active/20 max', np.nanmax(np.abs(dep - rec)), X.loc[np.nanargmax(np.abs(dep - rec))].to_dict())

# reference-proxy bounds (assumed node-vs-reference error structures; the network PM2.5 record is not available)
act = active(0.25); mon = days.month.values; winter = np.isin(mon, [11, 12, 1, 2, 3])
fill = np.zeros((len(days), 20), bool)
for k in range(20):
    fr = FR[:, k]; fill[:, k] = VAL[:, k] & (fr >= 0.25) & (fr < 0.75)
rows = []
for s, col in SITES.items():
    c = ref[col].values
    def run(cnode, lab):
        # cnode: (days x 20) concentration the node would measure; truth = mean over all commissioned node-days of cnode
        m = ~np.isnan(c); cn = cnode[m]; a = act[m]; v = VAL[m]
        true_pool = (cn * v).sum() / v.sum(); pooled = (cn * a).sum() / a.sum()
        dm = np.array([cn[i, a[i] == 1].mean() if a[i].any() else np.nan for i in range(len(cn))]); dt = np.array([cn[i, v[i]].mean() if v[i].any() else np.nan for i in range(len(cn))])
        ok = ~np.isnan(dm); daily = 100 * (np.nanmean(dm[ok]) - np.nanmean(dt)) / np.nanmean(dt)
        nb = []
        for k in range(20):
            vk = v[:, k]; on = a[:, k] == 1
            if on.any(): nb.append(100 * (cn[on, k].mean() - cn[vk, k].mean()) / cn[vk, k].mean())
        rows.append(dict(site=s, scenario=lab, pooled=100 * (pooled - true_pool) / true_pool, daily1=daily, node_median=np.median(nb)))
    base = np.repeat(c[:, None], 20, 1)
    run(base, 'Reference proxy (as in Eq. 15)')
    for g in (0.2, 0.5): run(base * (1 + g), f'Constant canyon increment +{int(100 * g)} %')
    for g in (0.2, 0.5): run(base * (1 + g * winter[:, None]), f'Winter-only canyon increment +{int(100 * g)} % (Nov–Mar)')
    for g in (0.2, 0.4): run(base * (1 + g * winter[:, None]), f'Humidity artefact of optical sensor +{int(100 * g)} % (Nov–Mar)')
    rng = np.random.default_rng(1)
    run(base * np.exp(0.3 * rng.standard_normal(base.shape) - 0.045), 'Random node error (log-sd 0.3)')
    run(base * (1 - 0.2 * fill), 'Partial-day nodes (25–75 % samples) read 20 % low')
    run(base * (1 + 0.2 * fill), 'Partial-day nodes (25–75 % samples) read 20 % high')
    other = ref[SITES['B' if s == 'A' else 'A']].values; run(np.repeat(np.where(np.isnan(other), c, other)[:, None], 20, 1) * (np.nanmean(c) / np.nanmean(other)), 'Seasonal cycle of the other site, rescaled')
P = pd.DataFrame(rows); P.to_csv('res_proxy_bounds.csv', index=False); print(P.round(1).to_string())
