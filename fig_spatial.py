import sys, numpy as np, pandas as pd; sys.path.insert(0,'.'); from style import *
R=pd.read_csv('res2_spatial_all.csv')
M=[('Hybrid: physics-prior residual (proposed)','prop','Physics-guided (proposed)'),('ML data-driven','ml','Data-driven ML'),
   ('Physics v2 calibrated prob.','phys','Physics prior'),('Persistence','pers','Persistence'),('Network persistence (others at origin)','clim','Network persistence')]
sets=[('Spatio-temporal, own calibration','Unseen node, own $s_n$'),('LONO-spatiotemporal','Unseen node, network-median $s_n$'),('Cold-start spatio-temporal','Unseen node, no own history')]
fig,ax=plt.subplots(1,3,figsize=(7.2,2.6),sharey=False)
for a,(st,title) in zip(ax,sets):
    g=R[R.setting==st]; H=sorted(g.h.unique())
    for name,k,lab in M:
        q=g[g.model==name].set_index('h')
        if q.empty: continue
        a.plot(H,q.loc[H].brier,color=C[k],ls=LS[k],marker=MK[k],ms=4,label=lab)
    a.set_title(title,fontsize=8); a.set_xticks(H); a.set_xlabel('Horizon h (d)')
ax[0].set_ylabel('Brier score (-)')
for a,l in zip(ax,'abc'): a.text(-0.2,1.05,'(%s)'%l,transform=a.transAxes,fontweight='bold')
hs,ls_=[],[]
for a in ax:
    for h_,l_ in zip(*a.get_legend_handles_labels()):
        if l_ not in ls_: hs.append(h_); ls_.append(l_)
fig.legend(hs,ls_,loc='lower center',ncol=5,fontsize=6.8,bbox_to_anchor=(0.5,-0.06))
fig.tight_layout(rect=(0,0.06,1,1)); fig.savefig('figs/fig5_spatial.png'); plt.close()
