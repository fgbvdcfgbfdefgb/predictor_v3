from pathlib import Path
import pandas as pd
bad=[]; n=0
for f in Path('data/processed').glob('*/*/*.parquet'):
 d=pd.read_parquet(f); n+=len(d)
 if not {'timestamp','open','high','low','close','volume'}<=set(d) or d.timestamp.duplicated().any(): bad.append(str(f))
print({'files':len(list(Path('data/processed').glob('*/*/*.parquet'))),'rows':n,'bad':bad[:20]})
raise SystemExit(bool(bad))
