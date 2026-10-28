import sys, numpy as np, pandas as pd; sys.path.insert(0,'.'); from style import *
R=pd.read_csv('res2_temporal.csv')
M=[('Hybrid: physics-prior residual (proposed)','prop','Physics-guided (proposed)'),('ML data-driven','ml','Data-driven ML'),
   ('Physics v2 assimilated (calibrated prob.)','phys','Physics prior (assimilated)'),('Persistence','pers','Persistence'),
   ('Hybrid: feature concatenation','concat','Feature-concatenation hybrid')]
fig,ax=plt.subplots(1,3,figsize=(7.2,2.6))
f=R[R.split=='fixed']; H=[1,7,14,28]
for name,k,lab in M:
    g=f[f.model==name].set_index('h').loc[H]
    ax[0].plot(H,g.brier,color=C[k],ls=LS[k],marker=MK[k],ms=4,label=lab)
r=R[R.split.str.startswith('roll')]
for name,k,lab in M:
    g=r[r.model==name].groupby('h').net_mae_pp.mean().loc[H]
    ax[1].plot(H,g.values,color=C[k],ls=LS[k],marker=MK[k],ms=4,label=lab)
rr=r[r.h==28]; months=sorted(rr.split.unique()); x=np.arange(len(months))
for name,k,lab in M[:4]:
    g=rr[rr.model==name].set_index('split').loc[months]
    ax[2].plot(x,g.net_mae_pp,color=C[k],ls=LS[k],marker=MK[k],ms=4,label=lab)
ax[2].set_xticks(x); ax[2].set_xticklabels([pd.Timestamp(m[5:]+'-01').strftime('%b') for m in months])
ax[0].set_xlabel('Horizon h (d)'); ax[0].set_ylabel('Brier score (-)'); ax[0].set_xticks(H); ax[0].set_title('Fixed split (Jan–Jun 2026)',fontsize=8)
ax[1].set_xlabel('Horizon h (d)'); ax[1].set_ylabel('Network availability MAE (pp)'); ax[1].set_xticks(H); ax[1].set_title('Rolling origin, mean of 8 months',fontsize=8)
ax[2].set_xlabel('Test month (2025–26)'); ax[2].set_ylabel('Network availability MAE (pp)'); ax[2].set_title('Rolling origin, h = 28 d',fontsize=8)
for a,l in zip(ax,'abc'): a.text(-0.2,1.05,'(%s)'%l,transform=a.transAxes,fontweight='bold')
h,l=ax[1].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=5,fontsize=6.8,bbox_to_anchor=(0.5,-0.06))
fig.tight_layout(rect=(0,0.06,1,1)); fig.savefig('figs/fig4_temporal.png'); plt.close()
