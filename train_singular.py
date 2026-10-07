"""Train the SINGULAR distribution (qqg, APS) models on a p-space dataset.  Training only (sample with generate.py).
Needs only model_singular.py and utils.py.  For the muon-decay (smooth, labelled) distribution use train_muon.py.

Usage:
    python train_singular.py --data-file TrainingData/SARGE_N10_xi40_1M.pt --output-dir runs/aps_xi40 [options]

Outputs in --output-dir:
    args.json        the command-line arguments
    training.log     epoch, loss
    loss.pdf         loss curve
    model.pt         raw + EMA weights, rewritten every 10 epochs and at the end (config included)
    Q0.pt            the q-space training data after the fixed embedding (N, n_particles, 3), saved before training
    metadata.pt      dict: T (number of diffusion steps), gammas (list of the T step sizes), N (training events), n_particles,
                     epsilon (regularisation of the reference drift, = the model's ref_eps)
    ckpts/epNNNN.pt  raw + EMA weights every --ckpt-every epochs (for evolution.py)
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
from dataclasses import fields  # noqa: E402

from model_singular import APS_B, APS_X, Config, DiffusionModel, embed_fixed, load_pspace  # noqa: E402


def parse_args(argv=None):
    d = argparse.Namespace(**{f.name: f.default for f in fields(Config)})   # the locked defaults (n_particles has none)
    p = argparse.ArgumentParser(description="Train the q-space diffusion model of record",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-file", required=True, help=".pt file with p-space events (N_events, N, 3)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-train", type=int, default=0, help="events used for training (0 = all events in the file)")
    p.add_argument("--seed", type=int, default=0)
    # embedding
    p.add_argument("--x", type=float, default=APS_X, help="scale of the fixed q-space embedding (default is 0.0846 for N = 10 APS events)")
    p.add_argument("--b", type=float, nargs=3, default=list(APS_B), help="boost of the fixed q-space embedding (default is (0,0,0)")
    # schedule
    p.add_argument("--t-steps", type=int, default=d.t_steps, help="total diffusion steps (incl. the geometric phase)")
    p.add_argument("--gamma-min", type=float, default=d.gamma_min, help="first step of the linear phase")
    p.add_argument("--gamma-max", type=float, default=d.gamma_max, help="last step of the linear phase")
    p.add_argument("--t-geom", type=int, default=d.t_geom, help="number of geometric small steps at the start")
    p.add_argument("--gamma-geom", type=float, default=d.gamma_geom, help="first geometric step (sigma_1 = sqrt(2 gamma))")
    p.add_argument("--gamma-geom-growth", type=float, default=d.gamma_geom_growth, help="growth factor of the geometric steps")
    p.add_argument("--ref-eps", type=float, default=d.ref_eps, help="regularization of the reference score")
    # training
    p.add_argument("--n-epochs", type=int, default=d.n_epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--ema-decay", type=float, default=d.ema_decay)
    p.add_argument("--ckpt-every", type=int, default=100, help="save ckpts/epNNNN.pt every K epochs (0 = off)")
    p.add_argument("--cache-dense-until", type=int, default=d.cache_dense_until, help="forward cache: store every step t <= this")
    p.add_argument("--cache-stride", type=int, default=d.cache_stride, help="... and every k-th step after")
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

    cfg = Config(n_particles=ps.shape[1],
                 t_steps=a.t_steps, gamma_min=a.gamma_min, gamma_max=a.gamma_max, t_geom=a.t_geom,
                 gamma_geom=a.gamma_geom, gamma_geom_growth=a.gamma_geom_growth, ref_eps=a.ref_eps,
                 batch_size=a.batch_size, n_epochs=a.n_epochs, lr=a.lr, ema_decay=a.ema_decay,
                 cache_dense_until=a.cache_dense_until, cache_stride=a.cache_stride, device_str=a.device)
    model = DiffusionModel(cfg, seed=a.seed)
    print(f"sigma_1 = {model.sigmas[1]:.2e}, sigma at the end of the geometric phase = {model.sigmas[cfg.t_geom]:.3f}, "
          f"sigma_T = {model.sigmas[-1]:.2f}, total diffusion time = {model.gammas.sum():.2f}", flush=True)

    # the q-space training data and the diffusion schedule, for use outside this pipeline
    # (model.forward_process(Q0, gammas), model.sample_fromQ_at_t(model.net, Q_t, t, gammas))
    torch.save(qs.cpu(), os.path.join(a.output_dir, "Q0.pt"))
    torch.save({"T": len(model.gammas), "gammas": model.gammas.cpu().tolist(),
                "N": int(qs.shape[0]), "n_particles": int(qs.shape[1]),
                "epsilon": float(cfg.ref_eps)}, os.path.join(a.output_dir, "metadata.pt"))

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
    ax.plot(losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel("ISM loss (sigma^2-weighted, Hutchinson)")
    ax.set_title(os.path.basename(os.path.normpath(a.output_dir)))
    ax.grid(True)
    fig.savefig(os.path.join(a.output_dir, "loss.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Done:", a.output_dir, flush=True)


if __name__ == "__main__":
    main()
