import argparse, os, random, json, yaml, numpy as np, torch
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP
from .data import discover, load_day, features
from .analyser import MarketAnalyser
from .model import ActorCritic


def setup(seed):
    rank = int(os.getenv("RANK", 0))
    world = int(os.getenv("WORLD_SIZE", 1))
    local = int(os.getenv("LOCAL_RANK", 0))
    if world > 1:
        torch.distributed.init_process_group(
            "nccl" if torch.cuda.is_available() else "gloo"
        )
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    dev = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
        torch.backends.cuda.matmul.allow_tf32 = True
    return rank, world, dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    c = yaml.safe_load(open(args.config))
    rank, world, dev = setup(c["seed"])
    days = discover(c["data_root"])
    if not days:
        raise SystemExit(
            "No complete days found. Expected "
            "data/processed/<exchange>/<symbol>/YYYY-MM-DD.parquet"
        )

    sample = load_day(next(iter(days.values())), c["symbols"])
    nf = features(sample, c["symbols"]).shape[1] + 1
    pc = c["ppo"]
    net = ActorCritic(
        nf,
        hidden=pc["hidden"],
        blocks=pc["blocks"],
        gradient_checkpointing=pc["gradient_checkpointing"],
    ).to(dev)
    parameter_count = net.parameter_count
    if abs(parameter_count - pc["target_parameters"]) > pc["parameter_tolerance"]:
        raise ValueError(
            f"Policy has {parameter_count:,} parameters; expected approximately "
            f"{pc['target_parameters']:,}"
        )
    model = DDP(net, device_ids=[dev.index], gradient_as_bucket_view=True) \
        if world > 1 and dev.type == "cuda" else net
    opt = torch.optim.AdamW(model.parameters(), lr=pc["lr"])
    amp = bool(pc["mixed_precision"] and dev.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    analyser = MarketAnalyser(c["analyser_trees"], c["seed"])
    out = Path(c["artifacts"])
    out.mkdir(exist_ok=True)
    metadata = {
        "trainable_parameters": parameter_count,
        "target_parameters": pc["target_parameters"],
        "world_size": world,
        "device": str(dev),
        "mixed_precision": amp,
        "architecture": "13-block residual MLP actor-critic",
    }
    if rank == 0:
        json.dump(metadata, open(out / "model_metadata.json", "w"), indent=2)
        print(json.dumps(metadata, indent=2))

    history = []
    for ep in range(c["epochs"]):
        day = random.choice(sorted(days))
        df = load_day(days[day], c["symbols"])
        F = features(df, c["symbols"])
        close = df[f"{c['symbols'][0]}_close"]
        analyser.fit(F.to_numpy(), close)
        pred, regime = analyser.feedback(F.to_numpy())
        X = np.c_[F.to_numpy(), pred].astype("float32")
        rets = np.nan_to_num(close.pct_change().to_numpy(), nan=0)
        steps = min(len(X) - 1, c["steps_per_epoch"])
        states = torch.from_numpy(X[:steps]).to(dev)

        # Rollout inference is detached: PPO recomputes only each small minibatch.
        model.eval()
        with torch.no_grad(), torch.autocast(
            device_type=dev.type, dtype=torch.float16, enabled=amp
        ):
            logits, values = model(states)
            dist = Categorical(logits=logits.float())
            actions = dist.sample()
            old = dist.log_prob(actions)
        model.train()

        rewards, equities, prev = [], [c["initial_cash"]], 0
        action_list = actions.cpu().tolist()
        for t, action in enumerate(action_list):
            pos = action - 3
            cost = abs(pos - prev) * (c["fee_bps"] + c["slippage_bps"]) / 1e4
            reward = (
                pos * rets[t + 1]
                - cost
                + pc["adviser_reward_weight"] * np.sign(pos) * pred[t]
            )
            rewards.append(float(reward))
            equities.append(equities[-1] * (1 + reward))
            prev = pos

        R, discounted = 0.0, []
        for reward in rewards[::-1]:
            R = reward + pc["gamma"] * R
            discounted.append(R)
        returns = torch.tensor(discounted[::-1], device=dev)
        advantage = returns - values.float()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        idx = torch.randperm(steps, device=dev)

        for _ in range(pc["update_epochs"]):
            for batch in idx.split(pc["batch_size"]):
                with torch.autocast(
                    device_type=dev.type, dtype=torch.float16, enabled=amp
                ):
                    new_logits, value = model(states[batch])
                    new_dist = Categorical(logits=new_logits.float())
                    logp = new_dist.log_prob(actions[batch])
                    ratio = (logp - old[batch]).exp()
                    s1 = ratio * advantage[batch]
                    s2 = ratio.clamp(1 - pc["clip"], 1 + pc["clip"]) * advantage[batch]
                    loss = (
                        -torch.min(s1, s2).mean()
                        + 0.5 * (value.float() - returns[batch]).pow(2).mean()
                        - pc["entropy_coef"] * new_dist.entropy().mean()
                    )
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

        advice = analyser.advice(pred, regime)
        metric = {
            "epoch": ep,
            "day": day,
            "return": equities[-1] / equities[0] - 1,
            "reward": sum(rewards),
            **advice,
        }
        history.append(metric)
        if rank == 0:
            fig, ax = plt.subplots(2, 1, figsize=(12, 7))
            ax[0].plot(equities)
            ax[0].set_title(f"Epoch {ep} | {day} | equity")
            ax[1].plot(np.asarray(action_list) - 3, alpha=0.7)
            ax[1].set_title("position (-3..3)")
            fig.suptitle(json.dumps(advice))
            fig.tight_layout()
            fig.savefig(out / f"epoch_{ep:05d}.png", dpi=130)
            plt.close(fig)
            json.dump(metric, open(out / f"epoch_{ep:05d}_advisor.json", "w"), indent=2)
            json.dump(history, open(out / "metrics.json", "w"), indent=2)
            if ep % c["checkpoint_every"] == 0:
                raw = model.module if hasattr(model, "module") else model
                torch.save(raw.state_dict(), out / f"policy_{ep:05d}.pt")

    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
