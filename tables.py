import pandas as pd, numpy as np
NAMES={'Persistence':'Persistence','Climatology':'Climatology','Physics v1 open-loop (s=1)':'Physics, original (open-loop, s = 1)',
 'Physics v2 open-loop (calibrated)':'Physics, calibrated (open-loop)','Physics v2 assimilated':'Physics, assimilated (load state)',
 'Physics v2 assimilated (calibrated prob.)':'Physics prior, Eq. (11)','ML data-driven':'Data-driven ML','Hybrid: feature concatenation':'Hybrid, feature concatenation',
 'Hybrid residual without monotone constraints':'Proposed without monotone constraints','Hybrid: physics-prior residual (proposed)':'**Proposed physics-guided learner**',
 'Network persistence (others at origin)':'Network persistence','Physics v2 open-loop':'Physics, calibrated (open-loop)','Physics v2 calibrated prob.':'Physics prior, Eq. (11)'}
def bold_best(df,cols_low,cols_high):
    out=df.copy().astype(object)
    for c in df.columns:
        v=df[c].astype(float); best=v.min() if c in cols_low else v.max()
        out[c]=[('**%.3f**' if abs(x-best)<1e-12 else '%.3f')%x if c.startswith(('BS','AUROC')) else (('**%.1f**' if abs(x-best)<1e-12 else '%.1f')%x) for x in v]
    return out
def md(df,index_name='Model'):
    cols=[index_name]+list(df.columns); s='| '+' | '.join(cols)+' |\n|:'+'-'*34+'|'+'|'.join(['-'*9+':']*len(df.columns))+'|\n'
    for i,r in df.iterrows(): s+='| '+NAMES.get(i,i)+' | '+' | '.join(str(x) for x in r.values)+' |\n'
    return s
def table2():
    R=pd.read_csv('res2_temporal.csv'); f=R[R.split=='fixed']; r=R[R.split.str.startswith('roll')]
    order=[k for k in NAMES if k in set(f.model)]
    T=pd.DataFrame(index=order)
    for h in (7,14,28):
        g=f[f.h==h].set_index('model'); T[f'BS h={h}']=g.brier; T[f'AUROC h={h}']=g.auroc
    T['$MAE_{net}$ h=28']=f[f.h==28].set_index('model').net_mae_pp
    T['BS h=28 (roll)']=r[r.h==28].groupby('model').brier.mean(); T['$MAE_{net}$ h=28 (roll)']=r[r.h==28].groupby('model').net_mae_pp.mean()
    T=T.loc[order]; low=[c for c in T if not c.startswith('AUROC')]; return md(bold_best(T,low,[]))
def table3():
    B=pd.read_csv('boot_temporal.csv'); vs={'Physics v2 assimilated (calibrated prob.)':'Physics prior','Persistence':'Persistence','ML data-driven':'Data-driven ML'}
    s='| Design | h (d) | Baseline | ΔBS [95% CI] | Δ$MAE_{\mathrm{net}}$ (pp) [95% CI] |\n|:--------|------:|:----------------|---------------------:|---------------------:|\n'
    for split in ('fixed','roll'):
        for h in (1,7,14,28):
            for b,lab in vs.items():
                q=B[(B.split==split)&(B.h==h)&(B.vs==b)]
                if q.empty: continue
                f=lambda m,d: (lambda r: ('%+.{0}f [%+.{0}f, %+.{0}f]'.format(d)%(r['diff'],r.lo,r.hi))+('*' if (r.lo>0 or r.hi<0) else ''))(q[q.metric==m].iloc[0])
                s+=f"| {'Fixed' if split=='fixed' else 'Rolling'} | {h} | {lab} | {f('brier',3)} | {f('net',1)} |\n"
    return s
if __name__=='__main__': print(table2()); print(table3())
def table4():
    R=pd.read_csv('res2_spatial_all.csv')
    sets=[('Spatio-temporal, own calibration','Unseen node, own $s_n$'),('LONO-spatiotemporal','Unseen node, median $s_n$'),('Cold-start spatio-temporal','Unseen node, no own history'),('LONO-history','Concurrent leave-one-node-out')]
    mods=['Persistence','Network persistence (others at origin)','Physics v2 calibrated prob.','ML data-driven','Hybrid: feature concatenation','Hybrid: physics-prior residual (proposed)']
    s='| Design | Model | BS h=7 | AUROC h=7 | BS h=28 | AUROC h=28 | $MAE_{\mathrm{net}}$ h=28 (pp) |\n|:-------------|:-------------------|-------:|-------:|-------:|-------:|-------:|\n'
    for st,lab in sets:
        g=R[R.setting==st]
        rows=[]
        for m in mods:
            q=g[g.model==m].set_index('h')
            if q.empty: continue
            rows.append((m,q.loc[7].brier,q.loc[7].auroc,q.loc[28].brier,q.loc[28].auroc,q.loc[28].net_mae_pp))
        bb7=min(r[1] for r in rows); bb28=min(r[3] for r in rows); ba7=max(r[2] for r in rows); ba28=max(r[4] for r in rows); bm=min(r[5] for r in rows)
        B=lambda v,b,f: ('**'+f%v+'**') if abs(v-b)<1e-12 else f%v
        for i,r in enumerate(rows):
            s+=f"| {lab if i==0 else ''} | {NAMES.get(r[0],r[0])} | {B(r[1],bb7,'%.3f')} | {B(r[2],ba7,'%.3f')} | {B(r[3],bb28,'%.3f')} | {B(r[4],ba28,'%.3f')} | {B(r[5],bm,'%.1f')} |\n"
    return s
def table5():
    R=pd.read_csv('res_repres.csv')
    src=[('Measured availability (deposit series)','Measured availability (full year)'),
         ('Proposed, h=28 d (Nov–Jun forecasts)','Physics-guided, 28-d forecast'),('Data-driven ML, h=28 d (Nov–Jun forecasts)','Data-driven ML, 28-d forecast'),
         ('Physics prior, h=28 d (Nov–Jun forecasts)','Physics prior, 28-d forecast'),('Persistence, h=28 d (Nov–Jun forecasts)','Persistence, 28-d forecast'),
         ('Proposed, h=7 d (Nov–Jun forecasts)','Physics-guided, 7-d forecast'),('Data-driven ML, h=7 d (Nov–Jun forecasts)','Data-driven ML, 7-d forecast'),
         ('Physics, 21 Wp panel (x2.1)','Physics counterfactual, 21 Wp panel'),('Unconstrained energy (all commissioned nodes up)','No energy limitation (commissioned nodes)')]
    s='| Availability source | Bias A, full year (%) | Bias B, full year (%) | Bias A, Nov–Jun (%) | Bias B, Nov–Jun (%) | Exceedance capture A (%) | Exceedance capture B (%) |\n|:----------------------------|-------:|-------:|-------:|-------:|-------:|-------:|\n'
    meas={s_:R[(R.source=='Proposed, h=28 d (Nov–Jun forecasts)')&(R.site==s_)].win_bias_meas.iloc[0] for s_ in 'AB'}
    for k,lab in src:
        a=R[(R.source==k)&(R.site=='A')].iloc[0]; b=R[(R.source==k)&(R.site=='B')].iloc[0]
        wa='%.1f'%a.win_bias_pred if pd.notna(a.get('win_bias_pred')) else ('%.1f'%meas['A'] if k.startswith('Measured') else '–')
        wb='%.1f'%b.win_bias_pred if pd.notna(b.get('win_bias_pred')) else ('%.1f'%meas['B'] if k.startswith('Measured') else '–')
        s+=f"| {lab} | {a.bias:.1f} | {b.bias:.1f} | {wa} | {wb} | {a.exc_capt:.1f} | {b.exc_capt:.1f} |\n"
    return s
