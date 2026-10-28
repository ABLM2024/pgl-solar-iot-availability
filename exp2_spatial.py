import sys, numpy as np, pandas as pd, time
sys.path.insert(0,'.'); import core, phys2 as P, feats2 as F, models as M
d=core.load(); days=d['days']; Tm=d['drv'].Tmean.values; Hc=d['drv'].Hclim.values
allm=np.ones(len(days),bool); dec=days<=pd.Timestamp('2025-12-31')
S_ALL=np.array([P.calibrate(d,k,allm,F.CFG,Hc,Tm) for k in range(20)])
S_DEC=np.array([P.calibrate(d,k,dec,F.CFG,Hc,Tm) for k in range(20)])
rows=[]; preds_all=[]; t0=time.time()
for setting,H,cold in (('LONO-history',(1,7,14,28),False),('LONO-spatiotemporal',(1,7,28),False),('Cold-start',(1,7,28),True)):
    Scal=S_DEC if setting=='LONO-spatiotemporal' else S_ALL
    for h in H:
        out=[]
        BASE=F.build(d,h,Scal,cold=cold)                                   # training nodes: own calibrated s
        for k in range(20):
            s_vec=Scal.copy(); s_vec[k]=np.median(np.delete(Scal,k))     # held-out node gets the median of the training nodes
            DF=pd.concat([BASE[BASE.k!=k],F.build(d,h,s_vec,cold=cold,only=[k])])
            if setting=='LONO-spatiotemporal': tr=DF[(DF.k!=k)&(DF.date<=pd.Timestamp('2025-12-31'))]; te=DF[(DF.k==k)&(DF.date>=pd.Timestamp('2026-01-01'))]
            else: tr=DF[DF.k!=k]; te=DF[DF.k==k]
            if len(te)==0: continue
            dfeat=F.COLD_DATA_F if cold else F.DATA_F
            p={'Network persistence (others at origin)':te.net_t.fillna(tr.y.mean()).values,
               'ML data-driven':M.gbm(tr,tr.y,te,dfeat),
               'Hybrid: feature concatenation':M.gbm(tr,tr.y,te,dfeat+F.PHYS_F,mono=F.MONO),
               'Hybrid: physics-prior residual (proposed)':M.residual_boost(tr,te,dfeat+F.PHYS_F,mono=F.MONO)}
            p['Physics v2 '+('open-loop' if cold else 'assimilated')]=te.up_sim.values
            p['Physics v2 calibrated prob.']=M.physics_prior(tr,te)[1]
            if not cold: p['Persistence']=te.act_t.values
            out.append(pd.DataFrame(dict(setting=setting,h=h,node=te.node.values,date=te.date.values,y=te.y.values,**p)))
        O=pd.concat(out); O.to_csv(f'p2_spatial_{setting}_h{h}.csv',index=False)
        for name in [c for c in O.columns if c not in ('setting','h','node','date','y')]:
            r=M.evaluate(O,O[name].values); r.update(setting=setting,h=h,model=name); rows.append(r)
        print(setting,h,round(time.time()-t0),flush=True)
R=pd.DataFrame(rows); R.to_csv('res2_spatial.csv',index=False)
pd.set_option('display.width',250)
for st in R.setting.unique():
    print(st); print(R[R.setting==st].pivot_table(index='model',columns='h',values=['brier','auroc','net_mae_pp']).round(3))
