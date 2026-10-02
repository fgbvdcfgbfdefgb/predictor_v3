from pathlib import Path
import random, pandas as pd, numpy as np
REQ=['timestamp','open','high','low','close','volume']
def discover(root):
 files=list(Path(root).glob('*/*/*.parquet'))
 by={}
 for f in files: by.setdefault(f.stem,[]).append(f)
 return {d:fs for d,fs in by.items() if len({x.parent.name for x in fs})>=3}
def load_day(files,symbols):
 frames=[]
 for s in symbols:
  cand=[f for f in files if f.parent.name==s]
  if not cand: raise ValueError(f'missing {s}')
  x=pd.read_parquet(random.choice(cand))[REQ].copy(); x['timestamp']=pd.to_datetime(x.timestamp,utc=True); x=x.set_index('timestamp').add_prefix(s+'_'); frames.append(x)
 df=pd.concat(frames,axis=1).sort_index().ffill().dropna()
 if len(df)<120: raise ValueError('day has fewer than 120 aligned rows')
 return df
def features(df,symbols):
 out=[]
 for s in symbols:
  c=df[f'{s}_close']; v=df[f'{s}_volume']
  out += [c.pct_change().rename(f'{s}_r1'),c.pct_change(5).rename(f'{s}_r5'),c.pct_change(30).rename(f'{s}_r30'),(c/c.rolling(60).mean()-1).rename(f'{s}_ma60'),(v/(v.rolling(60).mean()+1e-9)-1).rename(f'{s}_vol')]
 return pd.concat(out,axis=1).replace([np.inf,-np.inf],0).fillna(0).astype('float32')
