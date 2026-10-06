"""Train the p-space DDPM comparison model (model_pspace_ddpm.py) on a p-space dataset.  Training only.

Usage:
    python train_pspace_ddpm.py --data-file datasets/SARGE_N10_xi30_1M.pt --output-dir runs/pspace_ddpm_xi30 [options]

Outputs in --output-dir:
    args.json        the command-line arguments
    training.log     epoch, loss
    loss.pdf         loss curve
    model.pt         raw + EMA weights and the schedule, rewritten every 10 epochs and at the end
    ckpts/epNNNN.pt  raw + EMA weights every --ckpt-every epochs
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataclasses import fields  # noqa: E402

from model_pspace_ddpm import Config, PSpaceDDPM, load_pspace  # noqa: E402


def parse_args(argv=None):
    d = argparse.Namespace(**{f.name: f.default for f in fields(Config)})
    p = argparse.ArgumentParser(description="Train the p-space DDPM (predict-the-noise) comparison model",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-file", required=True, help=".pt file with p-space events (N_events, N, 3)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-train", type=int, default=0, help="events used for training (0 = all events in the file)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--x", type=float, default=d.x, help="the model is trained on P / x")
    # schedule (beta_t = 2 gamma_t; defaults = the schedule of model_singular)
    p.add_argument("--t-steps", type=int, default=d.t_steps, help="total diffusion steps (incl. the geometric phase)")
    p.add_argument("--gamma-min", type=float, default=d.gamma_min, help="first step of the linear phase")
    p.add_argument("--gamma-max", type=float, default=d.gamma_max, help="last step of the linear phase")
    p.add_argument("--t-geom", type=int, default=d.t_geom, help="number of geometric small steps at the start")
    p.add_argument("--gamma-geom", type=float, default=d.gamma_geom, help="first geometric step")
    p.add_argument("--gamma-geom-growth", type=float, default=d.gamma_geom_growth, help="growth factor of the geometric steps")
    # training
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
    ps = ps / a.x
    print(f"Training data: {tuple(ps.shape)}, rescaled by 1/x = 1/{a.x}: component std {ps.std():.4f}", flush=True)

    cfg = Config(n_particles=ps.shape[1], x=a.x, data_std=float(ps.std()),
                 t_steps=a.t_steps, gamma_min=a.gamma_min, gamma_max=a.gamma_max, t_geom=a.t_geom,
                 gamma_geom=a.gamma_geom, gamma_geom_growth=a.gamma_geom_growth,
                 batch_size=a.batch_size, n_epochs=a.n_epochs, lr=a.lr, ema_decay=a.ema_decay, device_str=a.device)
    model = PSpaceDDPM(cfg, seed=a.seed)
    print(f"sqrt(1-abar_1) = {model.sqrt_1m_abar[1]:.2e}, sqrt(abar_T) = {model.sqrt_abar[-1]:.2e} "
          f"(residual signal ~ {model.sqrt_abar[-1] * ps.std():.1e} vs prior std 1)", flush=True)

    logf = open(os.path.join(a.output_dir, "training.log"), "a")

    def cb(epoch, loss):
        logf.write(f"epoch={epoch + 1} loss={loss:.6f}\n")
        logf.flush()
        if a.ckpt_every > 0 and (epoch + 1) % a.ckpt_every == 0:
            os.makedirs(os.path.join(a.output_dir, "ckpts"), exist_ok=True)
            model.save(os.path.join(a.output_dir, "ckpts", f"ep{epoch + 1:04d}.pt"))

    losses = model.train(ps, seed=a.seed, callback=cb, ckpt_path=os.path.join(a.output_dir, "model.pt"))
    model.save(os.path.join(a.output_dir, "model.pt"))
    logf.close()

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel("DDPM loss |eps_theta - eps|^2")
    ax.set_title(os.path.basename(os.path.normpath(a.output_dir)))
    ax.grid(True)
    fig.savefig(os.path.join(a.output_dir, "loss.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Done:", a.output_dir, flush=True)


if __name__ == "__main__":
    main()
