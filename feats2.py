import numpy as np, pandas as pd, sys, copy
sys.path.insert(0,'.'); import core, phys2 as P
L=core.L
import os
CFG=P.Cfg(dod=float(os.environ.get('PGL_DOD','0.9')),R=float(os.environ.get('PGL_R','0.5')),latch_days=int(os.environ.get('PGL_NL','3')),T_latch=float(os.environ.get('PGL_TL','5.0')),driver='Hclim')   # selected on Jun-Dec 2025 only
DATA_F=['act_t','a_t','act_m7','act_m14','run_len','net_t','net_m7','net_chg7','dT_t','dT_m7','Tmean_t','Tmin_t','Tmean_m7',
        'ntemp_t','doy_sin','doy_cos','m_pole','m_lamp','m_bldg','age']
COLD_DATA_F=[f for f in DATA_F if f not in ('act_t','a_t','act_m7','act_m14','run_len')]
PHYS_F=['s_n','soc_t','on_t','off_days_t','latched_t','soc_h','soc_min','up_sim','ndown_sim','cum_net','Hclim_h','net_clim_h','cap_h','acc_h']
MONO={'s_n':1,'soc_t':1,'on_t':1,'off_days_t':-1,'latched_t':-1,'soc_h':1,'soc_min':1,'up_sim':1,'ndown_sim':-1,'cum_net':1,
      'Hclim_h':1,'net_clim_h':1,'cap_h':1,'acc_h':1}
def states(d,k,s,assim=True):
    """causal state trajectory: state after day t uses observations up to t only"""
    drv=d['drv']; H=drv[CFG.driver].values; Tm=drv.Tmean.values; n=len(H); st=int(np.argmax(d['valid'][:,k]))
    S=[None]*n; state=None
    for t in range(st,n):
        _,up,state=P.run(H,Tm,s,CFG,t,t+1,obs=d['act'][:,k] if assim else None,cm=d['cm'],state=state)
        S[t]=dict(state,up=bool(up[t]))
    return S
def propagate(state,t,h,s,Hc,Tc):
    x=dict(state); socs=[]; ups=[]; cum=0.0
    sf,up,_=P.run(Hc,Tc,s,CFG,t+1,t+h+1,state=x)
    for j in range(t+1,t+h+1): cum+=core.PANEL*s*Hc[j]*L.PR-24
    return sf[t+h], np.nanmin(sf[t+1:t+h+1]), float(up[t+h]), int((~up[t+1:t+h+1]).sum()), cum
def build(d,h,s_vec,cold=False,only=None):
    days=d['days']; n=len(days); drv=d['drv']; mo=days.month.values-1
    Hc=drv.Hclim.values; Tc=np.array(L.SITES['44N']['T'])[mo]+2.0; Tm=drv.Tmean.values
    act=d['act'].copy(); valid=d['valid']; cm=d['cm']; A=d['A'].copy()
    for t in np.where(cm)[0]: act[t]=act[t-1]; A[t]=A[t-1]
    roll=lambda x,w: pd.Series(x).rolling(w,min_periods=1).mean().values
    dT=drv.dT.values; dTf=pd.Series(dT).ffill().values; rows=[]
    for k in (range(act.shape[1]) if only is None else only):
        s=s_vec[k]; st=int(np.argmax(valid[:,k])); S=states(d,k,s,assim=not cold)
        others=[j for j in range(act.shape[1]) if j!=k]; vo=valid[:,others]
        net=np.where(vo.sum(1)>0,(act[:,others]*vo).sum(1)/np.maximum(vo.sum(1),1),np.nan)
        a_k=act[:,k]; rl=np.zeros(n)
        for t in range(1,n): rl[t]=rl[t-1]+1 if a_k[t]==a_k[t-1] else 0
        am7=roll(a_k,7); am14=roll(a_k,14); nm7=roll(net,7)
        for t in range(st,n-h):
            th=t+h
            if not valid[th,k] or cm[th]: continue
            soc_h,smin,up_h,nd,cum=propagate(S[t],t,h,s,Hc,Tc); doy=days[th].dayofyear
            rows.append(dict(node=d['nodes'][k],k=k,t=t,th=th,date=days[th],y=d['act'][th,k],
                act_t=a_k[t],a_t=A[t,k],act_m7=am7[t],act_m14=am14[t],run_len=min(rl[t],60),net_t=net[t],net_m7=nm7[t],
                net_chg7=net[t]-(net[t-7] if t>=7 else net[t]),dT_t=dT[t],dT_m7=np.nanmean(dTf[max(0,t-6):t+1]),
                Tmean_t=Tm[t],Tmin_t=drv.Tmin.values[t],Tmean_m7=Tm[max(0,t-6):t+1].mean(),ntemp_t=drv.n_temp_nodes.values[t],
                doy_sin=np.sin(2*np.pi*doy/365.25),doy_cos=np.cos(2*np.pi*doy/365.25),
                m_pole=float(d['mount'][k]=='pole'),m_lamp=float(d['mount'][k]=='lamp'),m_bldg=float(d['mount'][k]=='building'),
                age=(days[t]-pd.Timestamp(d['com'][k])).days,
                s_n=s,soc_t=S[t]['soc']/P.capw(Tm[t],CFG.dod),on_t=float(S[t]['on']),off_days_t=min(S[t]['off_days'],60),
                latched_t=float(S[t]['latched']),soc_h=soc_h,soc_min=smin,up_sim=up_h,ndown_sim=nd,cum_net=cum,
                Hclim_h=Hc[th],net_clim_h=core.PANEL*s*Hc[th]*L.PR-24,cap_h=P.capw(Tc[th],CFG.dod)/(84*CFG.dod),acc_h=L.acc_factor(Tc[th]+2)))
    return pd.DataFrame(rows)
