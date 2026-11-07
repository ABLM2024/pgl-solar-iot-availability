import numpy as np, pandas as pd, sys
import os
D=os.environ.get('PGL_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))+'/'
sys.path.insert(0,D+'D_code'); import leadacid as L
PANEL=10.0; UB25=84.0*L.DOD; LOAD=1.0; CF25=float(L.cap_factor(25.0))
# ------------------------------------------------------------------ data
def load():
    m=pd.read_csv(D+'F_siting/node_siting.csv')
    A=pd.read_csv(D+'B_availability/node_availability_daily.csv',parse_dates=['date']).set_index('date')[m.telemetry_id]
    A.columns=m.sensor_no.values
    drv=pd.read_csv('drivers.csv',parse_dates=['date']).set_index('date')
    days=A.index; com=pd.to_datetime(m.commissioned).values
    valid=np.array([[d>=c for c in com] for d in days])                  # commissioned
    act=(A.values>=float(os.environ.get('PGL_THR','0.25'))).astype(float)
    cnt=(act*valid).sum(1); prev=pd.Series(cnt).shift(1).values
    # common-mode (network-wide, non-energy) days: >=60 % drop vs previous day, recovery to >=80 % within 3 days
    cm=np.zeros(len(days),bool)
    for i in range(1,len(days)):
        if prev[i]>=5 and cnt[i]<=0.4*prev[i]:
            j=i
            while j<len(days) and j<i+4 and cnt[j]<0.8*prev[i]: j+=1
            if j<i+4 and j<len(days): cm[i:j]=True
    mount=m.mounting_type.str.split().str[0].map({'Dedicated':'pole','Lamp':'lamp','Building':'building'}).values
    return dict(days=days,A=A.values,act=act,valid=valid,cm=cm,drv=drv,mount=mount,sky=m.sky_exposure.values,nodes=m.sensor_no.values,com=com)
# ------------------------------------------------------------------ physics
def cap(Tm): return UB25*L.cap_factor(Tm)/CF25
def step(soc,H,Tm,s,load=LOAD):
    c=cap(Tm); soc=np.minimum(soc,c); net=PANEL*s*H*L.PR-24*load
    chg=np.where(net>0,net*L.ETA_RT*L.acc_factor(Tm+2.0),0.0)
    new=np.where(net>0,np.minimum(soc+chg,c),soc+net); up=new>=0
    return np.where(up,new,0.0),up,net,c
def open_loop(d,s,start_idx,H,Tm):
    """simulate one node from commissioning; returns soc, up"""
    n=len(H); soc=np.full(n,np.nan); up=np.zeros(n,bool); x=cap(Tm[start_idx])
    for t in range(start_idx,n):
        x,u,_,_=step(x,H[t],Tm[t],s); soc[t]=x; up[t]=u
    return soc,up
def calibrate_s(d,k,train_mask,grid=np.round(np.arange(0.3,2.51,0.1),2)):
    """per-node harvest multiplier (shading/orientation/panel ageing), fitted on training days only"""
    H=d['drv'].Hest.values; Tm=d['drv'].Tmean.values; st=int(np.argmax(d['valid'][:,k]))
    m=train_mask&d['valid'][:,k]&~d['cm']
    if m.sum()<10: return np.nan
    y=d['act'][:,k][m]; best=None
    for s in grid:
        _,up=open_loop(d,s,st,H,Tm); p=up[m]
        tpr=(p[y==1]==1).mean() if (y==1).any() else np.nan; tnr=(p[y==0]==0).mean() if (y==0).any() else np.nan
        ba=np.nanmean([tpr,tnr]); key=(ba,-abs(s-1.0))
        if best is None or key>best[0]: best=(key,s)
    return best[1]
def assimilate(d,k,s,H,Tm):
    """observation-corrected state: reported-down day -> empty reserve; reported-up while model empty -> 20 % reserve"""
    n=len(H); st=int(np.argmax(d['valid'][:,k])); soc=np.full(n,np.nan); x=cap(Tm[st])
    for t in range(st,n):
        x,u,_,c=step(x,H[t],Tm[t],s)
        if not d['cm'][t]:
            if d['act'][t,k]==0: x=0.0
            elif x<0.2*c: x=0.2*c
        soc[t]=x
    return soc
def propagate(soc0,t,h,s,Hc,Tc):
    """forecast h days ahead from state soc0 at day t with climatological drivers (no future weather used)"""
    x=soc0; mins=1e9; ndown=0; cum=0.0; dte=h+1
    for j in range(1,h+1):
        x,u,net,c=step(x,Hc[t+j],Tc[t+j],s); mins=min(mins,x/c); ndown+=(~u); cum+=net
        if (~u) and dte==h+1: dte=j
    return x/c, mins, float(u), ndown, cum, dte
