"""Train the muon-decay model of record (model_muon.py) on a p-space dataset.  Training only (sample with generate.py).
Needs only model_muon.py and utils.py.

Usage:
    python train_muon.py --data-file TrainingData/muon_500k.pt --output-dir runs/muon [options]

Outputs (same layout as train_singular.py): args.json, training.log, loss.pdf, model.pt (raw + EMA weights + schedule), ckpts/epNNNN.pt,
Q0.pt (embedded q-space training data), metadata.pt (T, gammas, N, n_particles, epsilon = 0).
Defaults = the muon model of record: paper MLP, paper linear schedule, paper loss, unregularised drift,
b = 0, x = 0.15, all events, 500 epochs, lr 3e-4, EMA 0.999.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_muon import MUON_B, MUON_X, MuonConfig, MuonDiffusionModel, embed_fixed, load_pspace  # noqa: E402


def parse_args(argv=None):
    d = MuonConfig()
    p = argparse.ArgumentParser(description="Train the muon-decay q-space diffusion model",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-file", required=True, help=".pt file with p-space events (N_events, 3, 3)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-train", type=int, default=0, help="events used for training (0 = all events in the file)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--x", type=float, default=MUON_X, help="scale of the fixed q-space embedding (default is 0.15)")
    p.add_argument("--b", type=float, nargs=3, default=list(MUON_B), help="boost of the fixed q-space embedding (0 0 0: no boost is default)")
    p.add_argument("--t-steps", type=int, default=d.t_steps)
    p.add_argument("--gamma-min", type=float, default=d.gamma_min)
    p.add_argument("--gamma-max", type=float, default=d.gamma_max)
    p.add_argument("--n-epochs", type=int, default=d.n_epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--ema-decay", type=float, default=d.ema_decay)
    p.add_argument("--ckpt-every", type=int, default=100, help="save ckpts/epNNNN.pt every K epochs (0 = off)")
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    os.makedirs(a.output_dir, exist_ok=True)
    print("Args:", vars(a), flush=True)
    with open(os.path.join(a.output_dir, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=2)
    ps = load_pspace(a.data_file, a.n_train)
    qs = embed_fixed(ps, b=tuple(a.b), x=a.x)
    print(f"Training data: {tuple(ps.shape)} -> q-space {tuple(qs.shape)}, |q| mean {qs.norm(dim=-1).mean():.3f}", flush=True)
    cfg = MuonConfig(n_particles=ps.shape[1], t_steps=a.t_steps, gamma_min=a.gamma_min, gamma_max=a.gamma_max,
                     batch_size=a.batch_size, n_epochs=a.n_epochs, lr=a.lr, ema_decay=a.ema_decay, device_str=a.device)
    model = MuonDiffusionModel(cfg, seed=a.seed)
    print(f"sigma_1 = {model.sigmas[1]:.2e}, sigma_T = {model.sigmas[-1]:.2f}, total diffusion time = {model.gammas.sum():.2f}", flush=True)
    torch.save(qs.cpu(), os.path.join(a.output_dir, "Q0.pt"))
    torch.save({"T": len(model.gammas), "gammas": model.gammas.cpu().tolist(), "N": int(qs.shape[0]), "n_particles": int(qs.shape[1]),
                "epsilon": 0.0}, os.path.join(a.output_dir, "metadata.pt"))
    logf = open(os.path.join(a.output_dir, "training.log"), "a")

    def cb(epoch, loss):
        logf.write(f"epoch={epoch + 1} loss={loss:.6f}\n")
        logf.flush()
        if a.ckpt_every > 0 and (epoch + 1) % a.ckpt_every == 0:
            os.makedirs(os.path.join(a.output_dir, "ckpts"), exist_ok=True)
            model.save(os.path.join(a.output_dir, "ckpts", f"ep{epoch + 1:04d}.pt"))

    losses = model.train(qs, seed=a.seed, callback=cb, ckpt_path=os.path.join(a.output_dir, "model.pt"))
    model.save(os.path.join(a.output_dir, "model.pt"))
    logf.close()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(losses); ax.set_xlabel("epoch"); ax.set_ylabel("ISM loss (time weighting, exact divergence)")
    ax.set_title(os.path.basename(os.path.normpath(a.output_dir))); ax.grid(True)
    fig.savefig(os.path.join(a.output_dir, "loss.pdf"), bbox_inches="tight"); plt.close(fig)
    print("Done:", a.output_dir, flush=True)


if __name__ == "__main__":
    main()
