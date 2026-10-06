"""Sample events from a trained model.  Sampling only: no plots, no comparison with the truth.

Usage:
    python generate.py --ckpt runs/aps_xi40/model.pt --n-samples 100000 --out runs/aps_xi40/generated.pt
                       [--weights ema|raw] [--seed 0] [--save-q] [--device cuda]

Loads the checkpoint with the model class named by its 'model' tag (model_singular.py or model_muon.py), runs the
reverse process (posterior-variance noise, noiseless last step) from the RAMBO prior, and writes the events to
--out as a torch tensor of p-space 3-momenta (n_samples, n_particles, 3), float32, total energy 1.  With --save-q the
q-space points are written next to it (<out stem>_q.pt).  Non-finite events (none expected) are dropped and reported.
Needs only model_singular.py / model_muon.py and utils.py.
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import qs_to_ps  # noqa: E402


def load_checkpoint(path, device="cuda", weights="ema"):
    """Same dispatch as sample.py: the checkpoint's 'model' tag selects the class (absent = singular)."""
    tag = torch.load(path, map_location="cpu", weights_only=False).get("model", "singular")
    if tag == "muon":
        from model_muon import MuonDiffusionModel
        return MuonDiffusionModel.load(path, device=device, weights=weights)
    from model_singular import DiffusionModel
    return DiffusionModel.load(path, device=device, weights=weights)


def main(argv=None):
    p = argparse.ArgumentParser(description="Sample events from a trained q-space diffusion model (no plots)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", required=True, help="model.pt or ckpts/epNNNN.pt written by train_singular.py / train_muon.py")
    p.add_argument("--out", required=True, help="output .pt file: p-space events (n_samples, n_particles, 3)")
    p.add_argument("--n-samples", type=int, default=100000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weights", choices=["ema", "raw"], default="ema", help="EMA weights (model of record) or raw weights")
    p.add_argument("--save-q", action="store_true", help="also write the q-space points to <out stem>_q.pt")
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)

    model = load_checkpoint(a.ckpt, device=a.device, weights=a.weights)
    print(f"loaded {a.ckpt}: {type(model).__name__}, N = {model.cfg.n_particles}, T = {len(model.gammas)}, {a.weights} weights", flush=True)
    t0 = time.time()
    Q = model.sample(a.n_samples, seed=a.seed)
    P = qs_to_ps(Q)
    ok = torch.isfinite(P).all(dim=(1, 2)) & torch.isfinite(Q).all(dim=(1, 2))
    if not ok.all():
        print(f"dropping {int((~ok).sum())} non-finite events", flush=True)
        Q, P = Q[ok], P[ok]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(P.cpu(), a.out)
    if a.save_q:
        torch.save(Q.cpu(), os.path.splitext(a.out)[0] + "_q.pt")
    print(f"sampled {len(P)} events in {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
