import sys, numpy as np, pandas as pd; sys.path.insert(0,'.'); from style import *
R=pd.read_csv('res_repres.csv')
fig,ax=plt.subplots(1,2,figsize=(7.2,2.7),gridspec_kw=dict(width_ratios=[1.25,1]))
# (a) bias in the Nov-Jun forecast window: predicted vs measured, h=28
src=[('Proposed, h=28 d (Nov–Jun forecasts)','prop','Physics-guided'),('Data-driven ML, h=28 d (Nov–Jun forecasts)','ml','Data-driven ML'),
     ('Physics prior, h=28 d (Nov–Jun forecasts)','phys','Physics prior'),('Persistence, h=28 d (Nov–Jun forecasts)','pers','Persistence')]
x=np.arange(2); w=0.16
meas=[R[(R.source==src[0][0])&(R.site==s)].win_bias_meas.iloc[0] for s in 'AB']
ax[0].bar(x-2.5*w,meas,w*0.92,color='#2b2b28',label='Measured availability')
for i,(s_,k,lab) in enumerate(src):
    v=[R[(R.source==s_)&(R.site==s)].win_bias_pred.iloc[0] for s in 'AB']
    ax[0].bar(x+(i-1.5)*w,v,w*0.92,color=C[k],label=lab+' (28-d forecast)')
ax[0].set_xticks(x); ax[0].set_xticklabels(['Site A (SPO-RO0065A)','Site B (SPO-RO0229A)']); ax[0].axhline(0,color='#8a8a85',lw=0.7)
ax[0].set_ylabel('Bias of availability-weighted mean (%)'); ax[0].set_title('Forecast window Nov 2025 – Jun 2026',fontsize=8)
ax[0].set_ylim(-32,1); ax[0].legend(fontsize=6.0,loc='upper center',bbox_to_anchor=(0.5,-0.13),ncol=3); ax[0].set_axisbelow(True)
# (b) full-year: measured vs physics counterfactuals
cf=[('Measured availability (deposit series)','#2b2b28','Measured'),('Physics, 21 Wp panel (x2.1)',C['phys'],'Physics, 21 Wp'),('Unconstrained energy (all commissioned nodes up)',C['clim'],'No energy limit')]
for i,(s_,c,lab) in enumerate(cf):
    v=[R[(R.source==s_)&(R.site==s)].exc_capt.iloc[0] for s in 'AB']
    ax[1].bar(x+(i-1)*0.25,v,0.23,color=c,label=lab)
    for j in range(2): ax[1].text(x[j]+(i-1)*0.25,v[j]+1.5,'%.0f'%v[j],ha='center',fontsize=6.3)
ax[1].set_xticks(x); ax[1].set_xticklabels(['Site A','Site B']); ax[1].set_ylabel('Exceedance days sampled (%)'); ax[1].set_ylim(0,125); ax[1].set_axisbelow(True)
ax[1].set_title('Full year, PM$_{2.5}$ > 15 µg m$^{-3}$',fontsize=8); ax[1].legend(fontsize=6.0,loc='upper center',bbox_to_anchor=(0.5,-0.13),ncol=3)
for a,l in zip(ax,'ab'): a.text(-0.2,1.07,'(%s)'%l,transform=a.transAxes,fontweight='bold')
fig.tight_layout(); fig.savefig('figs/fig6_repres.png'); plt.close()
