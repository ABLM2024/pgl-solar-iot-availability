import numpy as np, pandas as pd, glob
def block_boot(df, a, b, B=2000, seed=0, metric='brier'):
    """weekly block bootstrap of the difference metric(a)-metric(b); negative favours a"""
    df=df.copy(); df['wk']=pd.to_datetime(df.date).dt.to_period('W').astype(str)
    if metric=='brier':
        df['da']=(df[a]-df.y)**2; df['db']=(df[b]-df.y)**2
        g=df.groupby('wk')[['da','db']].agg(['sum','count'])
        sa=g[('da','sum')].values; sb=g[('db','sum')].values; n=g[('da','count')].values
        est=(sa.sum()-sb.sum())/n.sum(); rng=np.random.default_rng(seed); W=len(n); bs=[]
        for _ in range(B):
            i=rng.integers(0,W,W); bs.append((sa[i].sum()-sb[i].sum())/n[i].sum())
    else:  # network availability MAE (pp)
        g=df.groupby('date').agg(y=('y','mean'),pa=(a,'mean'),pb=(b,'mean')).reset_index()
        g['wk']=pd.to_datetime(g.date).dt.to_period('W').astype(str)
        g['ea']=100*np.abs(g.pa-g.y); g['eb']=100*np.abs(g.pb-g.y)
        w=g.groupby('wk')[['ea','eb']].agg(['sum','count'])
        sa=w[('ea','sum')].values; sb=w[('eb','sum')].values; n=w[('ea','count')].values
        est=(sa.sum()-sb.sum())/n.sum(); rng=np.random.default_rng(seed); W=len(n); bs=[]
        for _ in range(B):
            i=rng.integers(0,W,W); bs.append((sa[i].sum()-sb[i].sum())/n[i].sum())
    lo,hi=np.percentile(bs,[2.5,97.5]); return est,lo,hi
if __name__=='__main__':
    P='Hybrid: physics-prior residual (proposed)'; rows=[]
    for split in ['fixed','roll']:
        for h in (1,7,14,28):
            fs=sorted(glob.glob(f'p2_temporal_{split}*_h{h}.csv'))
            if not fs: continue
            df=pd.concat([pd.read_csv(f) for f in fs])
            for b in ['ML data-driven','Persistence','Physics v2 assimilated (calibrated prob.)']:
                for m in ['brier','net']:
                    e,lo,hi=block_boot(df,P,b,metric=m); rows.append(dict(split=split,h=h,vs=b,metric=m,diff=e,lo=lo,hi=hi))
    R=pd.DataFrame(rows); R.to_csv('boot_temporal.csv',index=False); pd.set_option('display.width',200); print(R.round(4).to_string())
