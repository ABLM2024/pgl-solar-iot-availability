"""Figures 3-6 and their numeric data."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.'); sys.path.insert(0, '/home/claude/apr/repo')
from style import *
import matplotlib.dates as mdates
import core, phys2 as P, feats2 as F
OUT = '/home/claude/apr/fig/'; DAT = '/home/claude/apr/res/figdata/'; import os; os.makedirs(DAT, exist_ok=True)
d = core.load(); days = d['days']; V = d['valid']
# ---------------- Fig 3: estimators
E = pd.read_csv('res_estimators.csv'); E = E[(E.thr == 0.25) & (E.limit == 15.0)].set_index('site')
N = pd.read_csv('res_nodecapture.csv')
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1.05, 1]))
labs = ['Daily network mean\n(≥ 1 node)', 'Daily network mean\n(≥ 2 nodes)', 'Pooled node-day\nmean, Eq. (1)', 'Per-node annual\nmeans (median)']
keys = ['daily1_bias', 'daily2_bias', 'pooled_bias', 'node_bias_median']; x = np.arange(4); rows = []
for j, (s, c, mk) in enumerate((('A', C['prop'], 'o'), ('B', C['ml'], 's'))):
    v = [E.loc[s, k] for k in keys]; lo = [E.loc[s, k + '_lo'] for k in keys]; hi = [E.loc[s, k + '_hi'] for k in keys]
    ax[0].errorbar(x + (j - 0.5) * 0.22, v, yerr=[np.array(v) - lo, np.array(hi) - v], fmt=mk, color=c, ms=5, capsize=2.5, lw=1.1, label=f'Site {s}')
    rows += [dict(site=s, estimator=k, bias=a, lo=b, hi=cc) for k, a, b, cc in zip(keys, v, lo, hi)]
ax[0].scatter(np.full(20, 3) - 0.11, N.bias_A, s=7, color=C['prop'], alpha=0.35, zorder=1); ax[0].scatter(np.full(20, 3) + 0.11, N.bias_B, s=7, color=C['ml'], alpha=0.35, zorder=1)
ax[0].axhline(0, color='#8a8a85', lw=0.8); ax[0].set_xticks(x); ax[0].set_xticklabels(labs, fontsize=6.6); ax[0].set_ylabel('Temporal-coverage bias (%)'); ax[0].legend(fontsize=7, loc='lower left')
for s, c, mk in (('A', C['prop'], 'o'), ('B', C['ml'], 's')):
    ax[1].scatter(N.capture_pct, N['bias_' + s], color=c, marker=mk, s=18, edgecolor='white', linewidth=0.4, label=f'Site {s}')
for xx in (75, 90): ax[1].axvline(xx, color='#8a8a85', lw=0.8, ls='--'); ax[1].text(xx + 0.8, -52, f'{xx}%', fontsize=6.5)
for nn in (1, 11, 18):
    r = N[N.node == nn].iloc[0]; ax[1].annotate(f'node {nn}', (r.capture_pct, r.bias_A), xytext=((-34 if nn == 1 else 4), (6 if nn != 18 else -12)), textcoords='offset points', fontsize=6.3)
ax[1].set_xlabel('Data capture of commissioned days (%)'); ax[1].set_ylabel('Bias of node annual mean (%)'); ax[1].legend(fontsize=7, loc='lower right')
for a, l in zip(ax, 'ab'): a.text(-0.14, 1.03, '(%s)' % l, transform=a.transAxes, fontweight='bold')
fig.tight_layout(); fig.savefig(OUT + 'fig3.png'); plt.close()
pd.DataFrame(rows).to_csv(DAT + 'fig3a.csv', index=False); N.to_csv(DAT + 'fig3b.csv', index=False)
# ---------------- Fig 4: physics as installed, multipliers, decomposition, panel sweep
Hc = d['drv'].Hclim.values; Tm = d['drv'].Tmean.values
S = pd.read_csv('res_s_profile.csv'); Sf = S[S.period == 'full'].set_index('node'); St = S[S.period == 'train'].set_index('node')
up_inst = np.zeros((len(days), 20), bool); up_tr = np.zeros((len(days), 20), bool); upv1 = np.zeros((len(days), 20), bool)
for k in range(20):
    st = int(np.argmax(V[:, k])); nd = int(d['nodes'][k])
    up_inst[:, k] = P.run(Hc, Tm, Sf.loc[nd, 's_best'], F.CFG, st, len(days))[1]; up_tr[:, k] = P.run(Hc, Tm, St.loc[nd, 's_best'], F.CFG, st, len(days))[1]
    upv1[:, k] = core.open_loop(d, 1.0, st, Hc, Tm)[1]
f7 = lambda U: pd.Series(np.nanmean(np.where(V, U, np.nan), 1), index=days).rolling(7, center=True, min_periods=1).mean()
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.2))
a = ax[0, 0]
for U, lab, c, ls in ((d['act'], 'Observed', C['obs'], '-'), (upv1, 'Original physics (s = 1, no LVD)', C['v1'], LS['v1']), (up_tr, 'Calibrated to Dec 2025, open-loop', C['phys'], LS['phys']),
                      (up_inst, 'As installed (full-year calibration)', C['prop'], '--')):
    a.plot(days, f7(U), color=c, ls=ls, lw=1.3 if lab != 'Observed' else 1.6, label=lab)
a.axvline(pd.Timestamp('2025-12-31'), color='#8a8a85', lw=0.8, ls='--'); a.set_ylabel('Active fraction of commissioned nodes\n(7-d mean)'); a.set_ylim(0, 1.05)
a.legend(fontsize=6, loc='lower left'); a.xaxis.set_major_formatter(mdates.DateFormatter('%b')); a.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
pd.DataFrame(dict(date=days, observed=f7(d['act']).values, original=f7(upv1).values, calibrated_train=f7(up_tr).values, as_installed=f7(up_inst).values)).to_csv(DAT + 'fig4a.csv', index=False)
a = ax[0, 1]; order = ['pole', 'lamp', 'building']
for g, (c, mk) in zip(order, [(C['prop'], 'o'), (C['ml'], 's'), (C['phys'], '^')]):
    sub = St[St.mount == g]; xs = order.index(g) + np.linspace(-0.15, 0.15, len(sub))
    a.vlines(xs, sub.near_lo, sub.near_hi, color=c, lw=1.0, alpha=0.7)
    cens = sub.censored_upper.values
    a.scatter(xs[~cens], sub.s_best[~cens], color=c, marker=mk, s=20, edgecolor='white', linewidth=0.5, zorder=3)
    if cens.any(): a.scatter(xs[cens], sub.s_best[cens], facecolor='white', edgecolor=c, marker=mk, s=26, zorder=3); [a.annotate('node 18, ≥ 2.0\n(censored)', (xx, 2.0), xytext=(-48, 8), textcoords='offset points', fontsize=6, arrowprops=dict(arrowstyle='-', lw=0.5)) for xx in xs[cens]]
    a.hlines(np.median(sub.s_best), order.index(g) - 0.25, order.index(g) + 0.25, color=C['obs'], lw=1.2)
a.set_xticks([0, 1, 2]); a.set_xticklabels(['Dedicated\npole', 'Lamp\npole', 'Building\nbracket']); a.set_ylabel('Effective energy-balance factor $s_n$ (-)')
St.reset_index().to_csv(DAT + 'fig4b.csv', index=False)
a = ax[1, 0]
# monthly down node-days by class (recomputed as in decomp.py)
import importlib.util
NET = np.zeros((len(days), 20))
for k in range(20): NET[:, k] = core.PANEL * Sf.loc[int(d['nodes'][k]), 's_best'] * Hc * core.L.PR - 24
cm = d['cm']; act = d['act'] * V; down = V & (act == 0) & ~cm[:, None]; energy = down & (NET < 0); maint = np.zeros_like(down); n = len(days)
for k in range(20):
    t = 0
    while t < n:
        if down[t, k]:
            j = t
            while j + 1 < n and (down[j + 1, k] or cm[j + 1]): j += 1
            if j + 1 < n and days[j + 1] in (pd.Timestamp('2026-03-18'), pd.Timestamp('2026-05-06')): maint[t:j + 1, k] = down[t:j + 1, k] & (NET[t:j + 1, k] >= 0)
            t = j + 1
        else: t += 1
other = down & ~energy & ~maint
assert energy.sum() == 1815 and maint.sum() == 717 and other.sum() == 149, (energy.sum(), maint.sum(), other.sum())
mon = pd.Series(days).dt.to_period('M').astype(str).values
M = pd.DataFrame(dict(month=mon, energy=energy.sum(1), maintenance=maint.sum(1), other=other.sum(1), common_mode=(V & cm[:, None]).sum(1))).groupby('month').sum()
b0 = np.zeros(len(M)); xm = np.arange(len(M))
for col, c, lab in (('energy', C['ml'], 'Energy deficit'), ('maintenance', C['prop'], 'Maintenance class (assumed)'), ('other', C['clim'], 'Other'), ('common_mode', C['v1'], 'Common mode')):
    a.bar(xm, M[col].values, bottom=b0, color=c, width=0.75, label=lab); b0 += M[col].values
a.set_xticks(xm); a.set_xticklabels([pd.Timestamp(m + '-01').strftime('%b') for m in M.index], fontsize=6.5); a.set_ylabel('Down node-days per month'); a.legend(fontsize=6, loc='upper left')
M.to_csv(DAT + 'fig4c.csv')
a = ax[1, 1]; Pn = pd.read_csv('res_panel.csv'); U = pd.read_csv('res_panel_unc.csv')
a.plot(Pn.Wp, Pn.djf_uptime, color=C['phys'], marker='o', ms=3.5, label='Calibrated model, with latch'); a.plot(Pn.Wp, Pn.djf_uptime_nolatch, color=C['pers'], ls=':', marker='D', ms=3, label='Without latch')
lo = U.djf_uptime.min(); hi = U.djf_uptime.max(); a.vlines(21, lo, hi, color=C['obs'], lw=2.2, alpha=0.6); a.text(22, lo - 6, f'21 Wp: {lo:.0f}–{hi:.0f}%\n(s, structure, ±15% winter H)', fontsize=6)
a.axhline(11.7, color=C['obs'], lw=0.8, ls='--'); a.text(30, 4, 'observed at 10 Wp: 11.7%', fontsize=6)
a.set_xlabel('Panel size (Wp)'); a.set_ylabel('DJF uptime of commissioned node-days (%)'); a.set_ylim(0, 103); a.legend(fontsize=6, loc='center right')
Pn.to_csv(DAT + 'fig4d.csv', index=False)
for a, l in zip(ax.ravel(), 'abcd'): a.text(-0.2, 1.03, '(%s)' % l, transform=a.transAxes, fontweight='bold')
fig.tight_layout(); fig.savefig(OUT + 'fig4.png'); plt.close()
# ---------------- Fig 5: rolling-origin skill per month and horizon
R = pd.read_csv('res3_temporal.csv'); R = R[R.split.str.startswith('roll')]
Mm = [('Proposed', 'prop', 'Physics-guided (proposed)'), ('ML data-driven', 'ml', 'Data-driven ML'), ('Physics prior', 'phys', 'Physics prior'), ('Persistence', 'pers', 'Persistence'),
      ('Discrete-time hazard model', 'concat', 'Discrete-time hazard')]
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True); months = sorted(R.split.unique()); xx = np.arange(len(months)); rows = []
for a, h in zip(ax, (7, 14, 28)):
    for name, k, lab in Mm:
        g = R[(R.model == name) & (R.h == h)].set_index('split').loc[months]
        a.plot(xx, g.brier, color=C[k], ls=LS[k], marker=MK[k], ms=3.5, label=lab); rows += [dict(h=h, model=name, month=m, brier=b) for m, b in zip(months, g.brier)]
    a.set_xticks(xx); a.set_xticklabels([pd.Timestamp(m[5:] + '-01').strftime('%b') for m in months], fontsize=6.5); a.set_title(f'h = {h} d', fontsize=8)
ax[0].set_ylabel('Brier score (-)')
for a, l in zip(ax, 'abc'): a.text(-0.12, 1.05, '(%s)' % l, transform=a.transAxes, fontweight='bold')
hh, ll = ax[0].get_legend_handles_labels(); fig.legend(hh, ll, loc='lower center', ncol=5, fontsize=6.8, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(OUT + 'fig5.png'); plt.close(); pd.DataFrame(rows).to_csv(DAT + 'fig5.csv', index=False)
# ---------------- Fig 6: bias forecasts with intervals; monthly errors
B = pd.read_csv('res_biasforecast.csv'); Mf = pd.read_csv('res_biasforecast_monthly.csv')
mods = ['Persistence', 'Physics prior', 'Data-driven ML', 'Proposed']; ck = dict(zip(mods, ['pers', 'phys', 'ml', 'prop']))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1, 1.25]))
a = ax[0]; ylab = []; y = 0
for h in (7, 28):
    for m in mods:
        for s, mk, off in (('A', 'o', -0.15), ('B', 's', 0.15)):
            r = B[(B.h == h) & (B.model == m) & (B.site == s)].iloc[0]
            a.errorbar(r.error, y + off, xerr=[[r.error - r.err_lo], [r.err_hi - r.error]], fmt=mk, color=C[ck[m]], ms=4, capsize=2, lw=1)
        ylab.append(f'{m}, {h} d'); y += 1
    y += 0.5
a.axvline(0, color='#8a8a85', lw=0.8); a.set_yticks([i for i in range(4)] + [i + 4.5 for i in range(4)]); a.set_yticklabels(ylab, fontsize=6.5); a.invert_yaxis()
a.set_xlabel('Error of forecast bias, Nov–Jun (pp)\n(circles site A, squares site B)')
a = ax[1]; ms_ = sorted(Mf.month.unique()); xm = np.arange(len(ms_))
for m in mods:
    g = Mf[(Mf.h == 28) & (Mf.model == m) & (Mf.site == 'A')].set_index('month').loc[ms_]
    a.plot(xm, g.err, color=C[ck[m]], ls=LS[ck[m]], marker=MK[ck[m]], ms=3.5, label=m if m != 'Proposed' else 'Physics-guided (proposed)')
a.axhline(0, color='#8a8a85', lw=0.8); a.set_xticks(xm); a.set_xticklabels([pd.Timestamp(m + '-01').strftime('%b') for m in ms_], fontsize=6.5)
a.set_ylabel('Monthly error of forecast bias,\nsite A, h = 28 d (pp)'); a.legend(fontsize=6.3, loc='upper left')
for a, l in zip(ax, 'ab'): a.text(-0.2 if l == 'b' else -0.55, 1.03, '(%s)' % l, transform=a.transAxes, fontweight='bold')
fig.tight_layout(); fig.savefig(OUT + 'fig6.png'); plt.close()
B.to_csv(DAT + 'fig6a.csv', index=False); Mf.to_csv(DAT + 'fig6b.csv', index=False)
print('figs ok')
