"""Physics-structure selection on the training half only (13 Jun - 31 Dec 2025)."""
import sys, numpy as np, pandas as pd, itertools
sys.path.insert(0,'.'); import core, phys2 as P
d=core.load(); days=d['days']; tr=days<=pd.Timestamp('2025-12-31')
rows=[]
for drv,dod,R,ld in itertools.product(['Hest','Hclim'],[0.5,0.7,0.9],[0.3,0.5,0.8],[3,7,14,999]):
    cfg=P.Cfg(dod=dod,R=R,latch_days=ld,driver=drv); H=d['drv'][drv].values; Tm=d['drv'].Tmean.values
    agree=[];bal=[]
    for k in range(20):
        s=P.calibrate(d,k,tr,cfg,H,Tm); st=int(np.argmax(d['valid'][:,k]))
        _,up,_=P.run(H,Tm,s,cfg,st,len(H)); m=tr&d['valid'][:,k]&~d['cm']; y=d['act'][m,k]; p=up[m]
        agree.append((p==y).mean()); bal.append(np.nanmean([(p[y==1]).mean() if (y==1).any() else np.nan,(~p[y==0]).mean() if (y==0).any() else np.nan]))
    rows.append(dict(driver=drv,dod=dod,R=R,latch=ld,agree=np.mean(agree),bacc=np.nanmean(bal)))
R=pd.DataFrame(rows).sort_values('bacc',ascending=False); R.to_csv('structure_selection.csv',index=False); print(R.head(15).to_string()); print(R.tail(5).to_string())
