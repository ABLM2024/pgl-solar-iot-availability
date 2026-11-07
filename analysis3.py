"""Pooled scores of the extended model set (exp3) for fixed and rolling designs; extra bootstrap comparisons. TAG='' or '_thr75'"""
import sys, glob, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
TAG = sys.argv[1] if len(sys.argv) > 1 else ''
def load(split, h):
    fs = sorted(glob.glob(f'p3_temporal{TAG}_{split}*_h{h}.csv')); return pd.concat([pd.read_csv(f, parse_dates=['date']) for f in fs])
rows = []
for split in ('fixed', 'roll'):
    for h in (7, 14, 28):
        df = load(split, h); models = [c for c in df.columns if c not in ('date', 'node', 'y', 'act_t')]
        for m in models:
            p = np.clip(df[m].values, 0, 1); y = df.y.values; g = df.groupby('date').agg(y=('y', 'mean'), p=(m, 'mean'))
            r = dict(split=split, h=h, model=m, brier=np.mean((p - y) ** 2), auroc=roc_auc_score(y, p), mae=100 * np.abs(g.p - g.y).mean())
            if split == 'roll':
                for lab, msk in (('noApr', df.date.dt.month != 4), ('Dec', df.date.dt.month == 12), ('Apr', df.date.dt.month == 4)):
                    dd = df[msk]; gg = dd.groupby('date').agg(y=('y', 'mean'), p=(m, 'mean'))
                    r['brier_' + lab] = np.mean((np.clip(dd[m], 0, 1) - dd.y) ** 2); r['mae_' + lab] = 100 * np.abs(gg.p - gg.y).mean()
            rows.append(r)
R = pd.DataFrame(rows); R.to_csv(f'res3_pooled{TAG}.csv', index=False)
pd.set_option('display.width', 250); print(R.round(3).to_string())
