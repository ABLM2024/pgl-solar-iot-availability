import sys, numpy as np, pandas as pd, time
sys.path.insert(0,'.'); import core, phys2 as P, feats2 as F, models as M
d=core.load(); days=d['days']; Tm=d['drv'].Tmean.values; Hc=d['drv'].Hclim.values
dec=days<=pd.Timestamp('2025-12-31'); CUT=pd.Timestamp('2025-12-31')
S_DEC=np.array([P.calibrate(d,k,dec,F.CFG,Hc,Tm) for k in range(20)])
rows=[]; t0=time.time()
for setting,cold,own in (('Cold-start spatio-temporal',True,False),('Spatio-temporal, own calibration',False,True)):
    for h in (1,7,28):
        out=[]; BASE=F.build(d,h,S_DEC,cold=cold)
        for k in range(20):
            s_vec=S_DEC.copy()
            if not own: s_vec[k]=np.median(np.delete(S_DEC,k))
            DF=pd.concat([BASE[BASE.k!=k],F.build(d,h,s_vec,cold=cold,only=[k])])
            tr=DF[(DF.k!=k)&(DF.date<=CUT)]; te=DF[(DF.k==k)&(DF.date>CUT)]
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
        O=pd.concat(out); O.to_csv(f'p2_spatial_{setting.replace(" ","_").replace(",","")}_h{h}.csv',index=False)
        for name in [c for c in O.columns if c not in ('setting','h','node','date','y')]:
            r=M.evaluate(O,O[name].values); r.update(setting=setting,h=h,model=name); rows.append(r)
        print(setting,h,round(time.time()-t0),flush=True)
R=pd.concat([pd.read_csv('res2_spatial.csv'),pd.DataFrame(rows)]); R.to_csv('res2_spatial_all.csv',index=False)
pd.set_option('display.width',250)
for st in [r['setting'] for r in rows][::6][:2]:
    print(st); print(pd.DataFrame(rows).query('setting==@st').pivot_table(index='model',columns='h',values=['brier','auroc','net_mae_pp']).round(3))
