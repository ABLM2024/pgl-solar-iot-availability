"""Daily physical drivers for the Pantelimon network (44.444 N, 26.163 E)."""
import numpy as np, pandas as pd, pvlib, sys
import os
D=os.environ.get('PGL_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))+'/'
sys.path.insert(0,D+'D_code'); import leadacid as L
LAT,LON=44.444,26.163; TILT,AZ=35.0,180.0
def clear_sky_inplane(days):
    loc=pvlib.location.Location(LAT,LON,tz='Europe/Bucharest',altitude=80)
    t=pd.date_range(days[0],days[-1]+pd.Timedelta('1D'),freq='10min',tz='Europe/Bucharest',inclusive='left')
    cs=loc.get_clearsky(t,model='ineichen'); sp=loc.get_solarposition(t)
    poa=pvlib.irradiance.get_total_irradiance(TILT,AZ,sp['apparent_zenith'],sp['azimuth'],cs['dni'],cs['ghi'],cs['dhi'],albedo=0.2)
    H=(poa['poa_global'].clip(lower=0)*(10/60)/1000).resample('D').sum()   # kWh m-2 d-1
    G=(cs['ghi']*(10/60)/1000).resample('D').sum()
    H.index=H.index.tz_localize(None); G.index=G.index.tz_localize(None)
    return H.reindex(days).values, G.reindex(days).values
def build():
    A=pd.read_csv(D+'B_availability/node_availability_daily.csv',parse_dates=['date']).set_index('date')
    days=A.index
    Hcs,Gcs=clear_sky_inplane(days)
    mo=days.month.values-1
    # climatological clearness: Table 2 in-plane monthly means / clear-sky in-plane monthly mean (computed over a full year)
    yr=pd.date_range('2025-01-01','2025-12-31',freq='D'); Hy,_=clear_sky_inplane(yr)
    Hcs_m=pd.Series(Hy,index=yr).groupby(yr.month).mean().values
    Kclim=np.array(L.SITES['44N']['H'])/Hcs_m
    T=pd.read_csv(D+'A_telemetry/Ambient_Temperature.csv',skiprows=1,parse_dates=['Time']).set_index('Time')
    g=T.groupby(T.index.normalize())
    tmax=g.max(); tmin=g.min(); tmean=g.mean(); nrd=g.count()
    ok=nrd>=6                                     # node-days with >= 6 of 8 readings
    dT=(tmax-tmin).where(ok).median(axis=1).reindex(days)
    Tn=tmin.where(ok).median(axis=1).reindex(days); Tm=tmean.where(ok).median(axis=1).reindex(days)
    nnodes=ok.sum(axis=1).reindex(days).fillna(0)
    clim_T=np.array(L.SITES['44N']['T'])[mo]
    Tm_f=Tm.fillna(pd.Series(clim_T+2.0,index=days)); Tn_f=Tn.fillna(pd.Series(clim_T-6+3.0,index=days))
    # Bristow-Campbell transmissivity shape, normalised to the monthly climatology
    dTbar=dT.groupby(days.month).transform('mean')
    B=0.036*np.exp(-0.154*dTbar); f=1-np.exp(-B*dT**2.4)
    r=(f/f.groupby(days.month).transform('mean')).clip(0.2,1.8).fillna(1.0)
    Hclim=Hcs*Kclim[mo]; Hest=Hclim*r.values
    out=pd.DataFrame(dict(Hcs=Hcs,Ghi_cs=Gcs,Kclim=Kclim[mo],Hclim=Hclim,r_bc=r.values,Hest=Hest,dT=dT.values,
                          Tmean=Tm_f.values,Tmin=Tn_f.values,T_obs=Tm.notna().values,n_temp_nodes=nnodes.values),index=days)
    return out
if __name__=='__main__':
    X=build(); X.to_csv('drivers.csv'); pd.set_option('display.width',200)
    print(X.resample('MS').mean().round(2)); print(X.describe().round(2).T)
