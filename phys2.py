"""Physics v2: VRLA energy balance with low-voltage disconnect (LVD), reconnect threshold and cold deep-discharge latch."""
import numpy as np, sys, os
RES=float(os.environ.get('PGL_RES','0.2'))
sys.path.insert(0,'.'); import core
L=core.L
if 'PGL_PR' in os.environ: L.PR=float(os.environ['PGL_PR'])
class Cfg:
    def __init__(s,dod=0.5,R=0.5,latch_days=7,T_latch=5.0,driver='Hest'): s.dod=dod; s.R=R; s.latch_days=latch_days; s.T_latch=T_latch; s.driver=driver
def capw(Tm,dod): return 84.0*dod*L.cap_factor(Tm)/core.CF25
def run(H,Tm,s,cfg,start,n,obs=None,cm=None,state=None,load=1.0):
    """returns per-day arrays soc_frac, up, and final state; obs (0/1) triggers assimilation if given"""
    if state is None: state=dict(soc=capw(Tm[start],cfg.dod),on=True,off_days=0,latched=False)
    soc,on,offd,lat=state['soc'],state['on'],state['off_days'],state['latched']
    socf=np.full(n,np.nan); up=np.zeros(n,bool)
    for t in range(start,n):
        c=capw(Tm[t],cfg.dod); soc=min(soc,c); gen=core.PANEL*s*H[t]*L.PR; acc=L.acc_factor(Tm[t]+2)
        if on:
            net=gen-24*load
            soc_new=min(soc+net*L.ETA_RT*acc,c) if net>0 else soc+net
            if soc_new<0: on=False; soc=0.0; offd=1; u=False
            else: soc=soc_new; u=True
        else:
            soc=min(soc+gen*L.ETA_RT*acc,c); offd+=1
            if offd>=cfg.latch_days and Tm[t]<cfg.T_latch: lat=True
            if (not lat) and soc>=cfg.R*c: on=True; offd=0
            u=False
        if obs is not None and not cm[t]:
            if obs[t]==1 and not u: on=True; lat=False; offd=0; soc=max(soc,RES*c); u=True
            elif obs[t]==0 and u: on=False; soc=0.0; offd=max(offd,1); u=False
        socf[t]=soc/c; up[t]=u
    return socf,up,dict(soc=soc,on=on,off_days=offd,latched=lat)
def calibrate(d,k,mask,cfg,H,Tm,grid=np.round(np.arange(0.3,2.51,0.1),2)):
    st=int(np.argmax(d['valid'][:,k])); n=len(H); m=mask&d['valid'][:,k]&~d['cm']
    if m.sum()<10: return 1.0
    y=d['act'][:,k][m]; best=None
    for s in grid:
        _,up,_=run(H,Tm,s,cfg,st,n); p=up[m]
        tpr=(p[y==1]).mean() if (y==1).any() else np.nan; tnr=(~p[y==0]).mean() if (y==0).any() else np.nan
        key=(np.nanmean([tpr,tnr]),-abs(s-1))
        if best is None or key>best[0]: best=(key,s)
    return best[1]
