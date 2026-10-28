import numpy as np, pandas as pd, lightgbm as lgb, warnings
from sklearn.metrics import roc_auc_score, brier_score_loss, balanced_accuracy_score, f1_score
warnings.filterwarnings('ignore')
P=dict(n_estimators=300,learning_rate=0.03,num_leaves=15,min_child_samples=40,subsample=0.8,subsample_freq=1,
       colsample_bytree=0.8,reg_lambda=1.0,verbose=-1)
def gbm(Xtr,ytr,Xte,feats,mono=None,seeds=5):
    if len(np.unique(ytr))<2: return np.full(len(Xte),float(ytr.mean()))
    out=np.zeros(len(Xte))
    for s in range(seeds):
        kw=dict(P,random_state=s)
        if mono: kw.update(monotone_constraints=[mono.get(f,0) for f in feats],monotone_constraints_method='advanced')
        m=lgb.LGBMClassifier(**kw).fit(Xtr[feats],ytr); out+=m.predict_proba(Xte[feats])[:,1]
    return out/seeds
def evaluate(df,p):
    y=df.y.values; r=dict(n=len(y),pos_rate=y.mean())
    r['auroc']=roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan
    r['brier']=brier_score_loss(y,np.clip(p,0,1)); r['bacc']=balanced_accuracy_score(y,p>=0.5) if len(np.unique(y))==2 else np.nan
    r['f1_down']=f1_score(1-y,(p<0.5).astype(int)) if (y==0).any() else np.nan
    g=pd.DataFrame(dict(date=df.date.values,y=y,p=p)).groupby('date').mean()
    r['net_mae_pp']=100*np.abs(g.p-g.y).mean()
    return r

from sklearn.linear_model import LogisticRegression
PRIOR_F=['up_sim','soc_h','soc_min','on_t','latched_t']
def physics_prior(tr,te):
    """calibrated physics probability: logistic link on physics-state outputs only (fitted on training)"""
    if len(np.unique(tr.y))<2: return np.full(len(tr),tr.y.mean()),np.full(len(te),tr.y.mean())
    c=LogisticRegression(C=1.0,max_iter=2000).fit(tr[PRIOR_F],tr.y)
    return c.predict_proba(tr[PRIOR_F])[:,1],c.predict_proba(te[PRIOR_F])[:,1]
def residual_boost(tr,te,feats,mono=None,seeds=5,n_est=150):
    """physics-prior residual boosting: trees start from logit(p_phys) and learn corrections"""
    ptr,pte=physics_prior(tr,te); lg=lambda p:np.log(np.clip(p,1e-4,1-1e-4)/(1-np.clip(p,1e-4,1-1e-4)))
    if len(np.unique(tr.y))<2: return pte
    out=np.zeros(len(te))
    for s in range(seeds):
        kw=dict(P,random_state=s,n_estimators=n_est)
        if mono: kw.update(monotone_constraints=[mono.get(f,0) for f in feats],monotone_constraints_method='advanced')
        m=lgb.LGBMClassifier(**kw).fit(tr[feats],tr.y,init_score=lg(ptr))
        out+=1/(1+np.exp(-(m.predict(te[feats],raw_score=True)+lg(pte))))
    return out/seeds
