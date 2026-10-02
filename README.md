# predictor_v3

Offline, multi-asset crypto research pipeline for BTC, ETH and LTC minute candles. It trains:

1. a **CPU market analyser** (regime + forward-return ensemble), and
2. a **450,043,592-parameter GPU-distributed PPO actor–critic** that consumes analyser feedback.

The policy uses 13 checkpointed residual MLP blocks at width 2,941. Mixed-precision training and PPO minibatches of 32 are enabled by default for 16 GB-class GPUs. DDP replicates the model on every GPU and distributes minibatch computation; it does not split model weights between GPUs.

Every epoch samples one random UTC day and all three assets. Metrics, equity/action graphs, and the adviser response are saved as PNG/JSON. This is research software—not financial advice or a guarantee of profitability.

## Data contract
Place Parquet files under `data/processed/<exchange>/<symbol>/<YYYY-MM-DD>.parquet`. Required columns: `timestamp,open,high,low,close,volume`; UTC timestamp, one-minute bars. The trainer never accesses the network.

## Snowflake / offline training
```bash
pip install -r requirements.txt
pip install -e .
python scripts/check_resources.py
python -m predictor_v3.train --config configs/default.yaml
```
The clone and pip installation may use the network; training itself only reads repository data. For 4 GPUs:
```bash
torchrun --standalone --nproc_per_node=4 -m predictor_v3.train --config configs/default.yaml
```

## Build dataset (maintainer machine only)
```bash
pip install -r requirements-data.txt
python scripts/download_data.py --start 2024-01-01 --end 2024-12-31 --exchanges binance kraken coinbase
```
Run `python scripts/validate_data.py`, then commit only a dataset size compatible with your Git/Snowflake limits. Full multi-exchange minute history is too large for ordinary Git; use Git LFS or a release archive only if your Snowflake clone workflow supports it.

## Safety
No live order execution, exchange credentials, leverage, or mining process control is included. Checkpoints use `state_dict` only. Dates are split chronologically to reduce leakage.
