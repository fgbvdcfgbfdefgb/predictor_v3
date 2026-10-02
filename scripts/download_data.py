import argparse,time
from pathlib import Path
import pandas as pd, ccxt
MAP={'binance':{'BTC':'BTC/USDT','ETH':'ETH/USDT','LTC':'LTC/USDT'},'kraken':{'BTC':'BTC/USD','ETH':'ETH/USD','LTC':'LTC/USD'},'coinbase':{'BTC':'BTC/USD','ETH':'ETH/USD','LTC':'LTC/USD'}}
ap=argparse.ArgumentParser(); ap.add_argument('--start',required=True); ap.add_argument('--end',required=True); ap.add_argument('--exchanges',nargs='+',default=list(MAP)); ap.add_argument('--root',default='data/processed'); a=ap.parse_args(); start=pd.Timestamp(a.start,tz='UTC'); end=pd.Timestamp(a.end,tz='UTC')+pd.Timedelta(days=1)
for exn in a.exchanges:
 ex=getattr(ccxt,exn)({'enableRateLimit':True})
 for asset,pair in MAP[exn].items():
  since=int(start.timestamp()*1000); rows=[]
  while since<int(end.timestamp()*1000):
   batch=ex.fetch_ohlcv(pair,'1m',since=since,limit=1000)
   if not batch: break
   rows.extend(batch); since=batch[-1][0]+60000; time.sleep(ex.rateLimit/1000)
  d=pd.DataFrame(rows,columns=['timestamp','open','high','low','close','volume']); d['timestamp']=pd.to_datetime(d.timestamp,unit='ms',utc=True); d=d[d.timestamp<end].drop_duplicates('timestamp');
  for day,g in d.groupby(d.timestamp.dt.strftime('%Y-%m-%d')):
   p=Path(a.root)/exn/asset/f'{day}.parquet'; p.parent.mkdir(parents=True,exist_ok=True); g.to_parquet(p,index=False,compression='zstd')
