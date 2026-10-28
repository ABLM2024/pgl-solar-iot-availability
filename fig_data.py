import sys, numpy as np, pandas as pd
sys.path.insert(0,'.'); from style import *; import core, phys2 as P, feats2 as F
import matplotlib.dates as mdates
d=core.load(); days=d['days']; drv=d['drv']
# ---------------- Fig 2: availability heatmap + network + drivers
fig,ax=plt.subplots(3,1,figsize=(7.2,6.0),gridspec_kw=dict(height_ratios=[2.2,1,1]),sharex=True)
Z=np.where(d['valid'],d['A'],np.nan).T
order=np.argsort([m for m in d['mount']],kind='stable')
cmap=plt.get_cmap('Blues').copy(); cmap.set_bad('#f2f2ef')
im=ax[0].imshow(Z[order],aspect='auto',cmap=cmap,vmin=0,vmax=1,interpolation='nearest',
    extent=[mdates.date2num(days[0]),mdates.date2num(days[-1]+pd.Timedelta(days=1)),20,0])
ax[0].set_yticks(np.arange(20)+0.5); ax[0].set_yticklabels(['%d (%s)'%(d['nodes'][i],{'pole':'P','lamp':'L','building':'B'}[d['mount'][i]]) for i in order],fontsize=6)
ax[0].grid(False); ax[0].set_ylabel('Node (mounting)')
cb=fig.colorbar(im,ax=ax[0],pad=0.01,fraction=0.025); cb.set_label('Daily availability (-)')
for t in days[d['cm']]: ax[0].axvline(mdates.date2num(t)+0.5,color='#e34948',lw=0.6,alpha=0.7)
net=np.nanmean(np.where(d['valid'],d['act'],np.nan),1)
ax[1].plot(days,np.where(d['valid'],d['act'],0).sum(1)/20,color=C['obs'],lw=1.2,label='Network availability (20-node basis)')
ax[1].plot(days,net,color=C['prop'],lw=1.0,ls='--',label='Active fraction of commissioned nodes')
ax[1].set_ylabel('Fraction'); ax[1].set_ylim(0,1.05); ax[1].legend(loc='lower left',fontsize=7)
ax[2].plot(days,drv.Hcs,color=C['clim'],lw=1.0,ls=':',label='Clear-sky in-plane $H_{cs}$')
ax[2].plot(days,drv.Hclim,color=C['phys'],lw=1.4,label='Climatological in-plane $H_{clim}$')
ax[2].plot(days,drv.Hest,color=C['ml'],lw=0.6,alpha=0.8,label='Temperature-range proxy $H_{BC}$')
ax[2].axhline(24/(10*0.65),color=C['obs'],lw=0.8,ls='--'); ax[2].text(days[5],24/6.5+0.15,'break-even at 1 W, 10 Wp',fontsize=6.5)
ax[2].set_ylabel('kWh m$^{-2}$ d$^{-1}$'); ax[2].legend(loc='upper center',ncol=3,fontsize=6.5)
ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y')); ax[2].xaxis.set_major_locator(mdates.MonthLocator())
for a,l in zip(ax,'abc'): a.text(-0.08,1.02,'(%s)'%l,transform=a.transAxes,fontweight='bold')
fig.savefig('figs/fig2_data.png'); plt.close()
# ---------------- Fig 3: physics v2 (structure selected on Jun-Dec 2025) vs observation, full-year calibration for illustration
Hc=drv.Hclim.values; Tm=drv.Tmean.values; tr=days<=pd.Timestamp('2025-12-31')
s_tr=np.array([P.calibrate(d,k,tr,F.CFG,Hc,Tm) for k in range(20)])
ups=np.zeros((len(days),20)); upa=np.zeros((len(days),20)); upv1=np.zeros((len(days),20))
for k in range(20):
    st=int(np.argmax(d['valid'][:,k])); ups[:,k]=P.run(Hc,Tm,s_tr[k],F.CFG,st,len(days))[1]
    upa[:,k]=P.run(Hc,Tm,s_tr[k],F.CFG,st,len(days),obs=d['act'][:,k],cm=d['cm'])[1]
    upv1[:,k]=core.open_loop(d,1.0,st,Hc,Tm)[1]
V=d['valid']; f=lambda U:pd.Series(np.nanmean(np.where(V,U,np.nan),1),index=days).rolling(7,center=True,min_periods=1).mean()
fig,ax=plt.subplots(1,2,figsize=(7.2,2.6),gridspec_kw=dict(width_ratios=[2.3,1]))
ax[0].plot(days,f(d['act']),color=C['obs'],lw=1.6,label='Observed')
ax[0].plot(days,f(upv1),color=C['v1'],ls=LS['v1'],label='Original physics (s = 1, no LVD)')
ax[0].plot(days,f(ups),color=C['phys'],ls=LS['phys'],label='Calibrated physics, open-loop (calibrated to Dec 2025)')
ax[0].axvline(pd.Timestamp('2025-12-31'),color='#8a8a85',lw=0.8,ls='--'); ax[0].text(pd.Timestamp('2026-01-04'),0.97,'calibration | test',fontsize=6.5,va='top')
ax[0].set_ylabel('Active fraction (7-d mean)'); ax[0].set_ylim(0,1.05); ax[0].legend(fontsize=6.5,loc='upper center',bbox_to_anchor=(0.5,-0.13),ncol=3)
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%b')); ax[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
for g,(c,mk) in zip(['pole','lamp','building'],[(C['prop'],'o'),(C['ml'],'s'),(C['phys'],'^')]):
    v=s_tr[d['mount']==g]; ax[1].scatter(np.full(len(v),['pole','lamp','building'].index(g))+np.linspace(-0.12,0.12,len(v)),v,color=c,marker=mk,s=22,edgecolor='white',linewidth=0.5,zorder=3)
    ax[1].hlines(np.median(v),['pole','lamp','building'].index(g)-0.25,['pole','lamp','building'].index(g)+0.25,color=C['obs'],lw=1.2)
ax[1].set_xticks([0,1,2]); ax[1].set_xticklabels(['Dedicated\npole','Lamp\npole','Building\nbracket']); ax[1].set_ylabel('Harvest multiplier $s_n$ (-)')
for a,l in zip(ax,'ab'): a.text(-0.12,1.02,'(%s)'%l,transform=a.transAxes,fontweight='bold')
fig.savefig('figs/fig3_physics.png'); plt.close()
pd.DataFrame(dict(node=d['nodes'],mount=d['mount'],s_train=s_tr)).to_csv('s_train.csv',index=False)
print(pd.DataFrame(dict(node=d['nodes'],mount=d['mount'],s=s_tr)).groupby('mount').s.describe())
