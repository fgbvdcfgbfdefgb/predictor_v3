import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
class MarketAnalyser:
 def __init__(self,trees=200,seed=42): self.model=ExtraTreesRegressor(n_estimators=trees,n_jobs=-1,random_state=seed,min_samples_leaf=5); self.ready=False
 def fit(self,X,close):
  y=np.nan_to_num(close.pct_change().shift(-5).to_numpy(),nan=0); self.model.fit(X[:-5],y[:-5]); self.ready=True
 def feedback(self,X):
  p=self.model.predict(X) if self.ready else np.zeros(len(X)); vol=np.std(X[:,::5],axis=1); regime=np.where(vol>np.median(vol),'high-vol','normal'); return p.astype('float32'),regime
 def advice(self,p,regime):
  score=float(np.mean(p)); return {"expected_5m_return":score,"regime":max(set(regime),key=list(regime).count),"stance":"risk-on" if score>0.0002 else "risk-off" if score<-.0002 else "neutral","confidence":float(min(1,abs(score)/.002))}
