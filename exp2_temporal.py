import sys, numpy as np, pandas as pd, time
sys.path.insert(0,'.'); import core, phys2 as P, feats2 as F, models as M
d=core.load(); days=d['days']; Tm=d['drv'].Tmean.values; Hc=d['drv'].Hclim.values
def run_split(cut, t0_, t1_, H=(1,7,14,28), tag=''):
    trmask=days<=cut
    s_cal=np.array([P.calibrate(d,k,trmask,F.CFG,Hc,Tm) for k in range(20)])
    res=[]
    # published-model analogue: original v1 physics, s=1, 50 % DoD, no LVD, climatological driver
    v1={}
    for k in range(20):
        st=int(np.argmax(d['valid'][:,k])); v1[k]=core.open_loop(d,1.0,st,Hc,Tm)[1]
    olc={}
    for k in range(20):
        st=int(np.argmax(d['valid'][:,k])); olc[k]=P.run(Hc,Tm,s_cal[k],F.CFG,st,len(Hc))[1]
    for h in H:
        DF=F.build(d,h,s_cal); tr=DF[DF.date<=cut]; te=DF[(DF.date>=t0_)&(DF.date<=t1_)]
        mrate=tr.groupby(tr.date.dt.month).y.mean(); clim=te.date.dt.month.map(mrate).fillna(tr.y.mean()).values
        _,pphys=M.physics_prior(tr,te)
        preds={'Persistence':te.act_t.values,'Climatology':clim,
               'Physics v1 open-loop (s=1)':np.array([v1[k][th] for k,th in zip(te.k,te.th)],float),
               'Physics v2 open-loop (calibrated)':np.array([olc[k][th] for k,th in zip(te.k,te.th)],float),
               'Physics v2 assimilated':te.up_sim.values,'Physics v2 assimilated (calibrated prob.)':pphys,
               'ML data-driven':M.gbm(tr,tr.y,te,F.DATA_F),
               'Hybrid: feature concatenation':M.gbm(tr,tr.y,te,F.DATA_F+F.PHYS_F,mono=F.MONO),
               'Hybrid: physics-prior residual (proposed)':M.residual_boost(tr,te,F.DATA_F+F.PHYS_F,mono=F.MONO),
               'Hybrid residual without monotone constraints':M.residual_boost(tr,te,F.DATA_F+F.PHYS_F)}
        for name,p in preds.items():
            r=M.evaluate(te,np.asarray(p,float)); r.update(model=name,h=h,split=tag); res.append(r)
        pd.DataFrame(dict(date=te.date.values,node=te.node.values,y=te.y.values,**{n:np.asarray(p,float) for n,p in preds.items()})).to_csv(f'p2_temporal_{tag}_h{h}.csv',index=False)
        print(tag,h,len(tr),len(te),flush=True)
    return res
t0=time.time()
R=run_split(pd.Timestamp('2025-12-31'),pd.Timestamp('2026-01-01'),pd.Timestamp('2026-06-13'),tag='fixed')
for M0 in pd.date_range('2025-11-01','2026-06-01',freq='MS'):
    R+=run_split(M0-pd.Timedelta(days=1),M0,M0+pd.offsets.MonthEnd(0),H=(1,7,14,28),tag='roll_'+M0.strftime('%Y-%m'))
R=pd.DataFrame(R); R.to_csv('res2_temporal.csv',index=False); print(time.time()-t0)
