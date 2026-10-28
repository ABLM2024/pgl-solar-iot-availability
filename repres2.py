import sys, numpy as np, pandas as pd, glob
sys.path.insert(0,'.'); import core, phys2 as P, feats2 as F
d=core.load(); days=d['days']; D=core.D
ref=pd.read_csv(D+'C_reference/reference_PM25_daily_EEA.csv',parse_dates=['date']).set_index('date').reindex(days)
cols={'A':'PM25_siteA_SPO_RO0065A','B':'PM25_siteB_SPO_RO0229A'}
def stats(a,c):
    m=~np.isnan(c)&~np.isnan(a); a=a[m]; c=c[m]; w=(a*c).sum()/a.sum(); tru=c.mean(); exc=c>15
    return dict(true=tru,weighted=w,bias=100*(w-tru)/tru,exc_days=int(exc.sum()),exc_capt=100*a[exc].sum()/exc.sum(),mean_avail=100*a.mean())
rows=[]
A_meas=ref.network_availability_fraction.values
for s,c in cols.items(): rows.append(dict(source='Measured availability (deposit series)',site=s,**stats(A_meas,ref[c].values)))
# my reconstruction on the 20-node basis from node series (check)
A_rec=(np.where(d['valid'],d['act'],0)).sum(1)/20
for s,c in cols.items(): rows.append(dict(source='Measured, active node-days / 20',site=s,**stats(A_rec,ref[c].values)))
# forecasts: rolling origin Nov-Jun; outside the forecast window the measured active fraction is used
for h in (7,28):
    df=pd.concat([pd.read_csv(f,parse_dates=['date']) for f in glob.glob(f'p2_temporal_roll_*_h{h}.csv')])
    for mdl,lab in (('Hybrid: physics-prior residual (proposed)','Proposed'),('ML data-driven','Data-driven ML'),('Physics v2 assimilated (calibrated prob.)','Physics prior'),('Persistence','Persistence')):
        g=df.groupby('date')[mdl].sum().reindex(days)/20; a=np.where(np.isnan(g.values),A_rec,g.values)
        win=~np.isnan(g.values)
        for s,c in cols.items():
            r=stats(a,ref[c].values); rw=stats(g.values[win],ref[c].values[win]); rm=stats(A_rec[win],ref[c].values[win])
            rows.append(dict(source=f'{lab}, h={h} d (Nov–Jun forecasts)',site=s,**r,win_bias_pred=rw['bias'],win_bias_meas=rm['bias']))
# physics counterfactuals (open-loop v2, multipliers calibrated on the full year, structure from Jun-Dec)
Hc=d['drv'].Hclim.values; Tm=d['drv'].Tmean.values; allm=np.ones(len(days),bool)
S=np.array([P.calibrate(d,k,allm,F.CFG,Hc,Tm) for k in range(20)])
for fac,lab in ((1.0,'Physics, as installed (10 Wp)'),(2.1,'Physics, 21 Wp panel (x2.1)'),(100.0,'Unconstrained energy (all commissioned nodes up)')):
    U=np.zeros((len(days),20))
    for k in range(20):
        st=int(np.argmax(d['valid'][:,k]))
        U[:,k]=P.run(Hc,Tm,S[k]*fac,F.CFG,st,len(days))[1] if fac<100 else d['valid'][:,k]
    a=(U*d['valid']).sum(1)/20
    for s,c in cols.items(): rows.append(dict(source=lab,site=s,**stats(a,ref[c].values)))
R=pd.DataFrame(rows); R.to_csv('res_repres.csv',index=False); pd.set_option('display.width',250); print(R.round(2).to_string())
