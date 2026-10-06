"""Train the q-space flow-matching comparison model with the RAMBO prior (model_qspace_fm.py).  Training only.

Usage:
    python train_qspace_fm.py --data-file datasets/SARGE_N10_xi30_1M.pt --output-dir runs/qspace_fm_xi30 [options]

Outputs in --output-dir:
    args.json        the command-line arguments
    training.log     epoch, loss
    loss.pdf         loss curve
    model.pt         raw + EMA weights, rewritten every 10 epochs and at the end
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

from model_qspace_fm import APS_B, APS_X, Config, QSpaceFlowMatching, embed_fixed, load_pspace  # noqa: E402


def parse_args(argv=None):
    d = argparse.Namespace(**{f.name: f.default for f in fields(Config)})
    p = argparse.ArgumentParser(description="Train the q-space flow-matching (RAMBO prior) comparison model",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-file", required=True, help=".pt file with p-space events (N_events, N, 3)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-train", type=int, default=0, help="events used for training (0 = all events in the file)")
    p.add_argument("--seed", type=int, default=0)
    # embedding
    p.add_argument("--x", type=float, default=APS_X, help="scale of the fixed q-space embedding")
    p.add_argument("--b", type=float, nargs=3, default=list(APS_B), help="boost of the fixed q-space embedding")
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
    qs = embed_fixed(ps, b=tuple(a.b), x=a.x)
    print(f"Training data: {tuple(ps.shape)} -> q-space {tuple(qs.shape)}, |q| mean {qs.norm(dim=-1).mean():.3f} "
          f"(RAMBO prior: 2)", flush=True)

    cfg = Config(n_particles=ps.shape[1], batch_size=a.batch_size, n_epochs=a.n_epochs, lr=a.lr,
                 ema_decay=a.ema_decay, device_str=a.device)
    model = QSpaceFlowMatching(cfg, seed=a.seed)

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
    ax.set_ylabel("flow-matching loss, Eq. (13)")
    ax.set_title(os.path.basename(os.path.normpath(a.output_dir)))
    ax.grid(True)
    fig.savefig(os.path.join(a.output_dir, "loss.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Done:", a.output_dir, flush=True)


if __name__ == "__main__":
    main()
