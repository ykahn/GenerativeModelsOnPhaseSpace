"""Sample events from a trained q-space flow-matching model (model_qspace_fm.py).  Sampling only: no plots, no comparison
with the truth.

Usage:
    python generate_qspace_fm.py --ckpt runs/qspace_fm_xi30/model.pt --n-samples 100000 --out runs/qspace_fm_xi30/generated.pt
                                 [--n-steps 500] [--weights ema|raw] [--seed 0] [--save-q] [--batch-size 100000] [--device cuda]

Integrates the learned ODE with --n-steps Euler steps from the RAMBO prior (t = 1) to t = 0, maps the q-space points to
p-space (utils.qs_to_ps: total energy 1, total momentum 0) and writes the 3-momenta (n_samples, n_particles, 3), float32,
to --out.  With --save-q the q-space points are written next to it (<out stem>_q.pt).  Non-finite events are dropped and
reported.
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_qspace_fm import QSpaceFlowMatching  # noqa: E402
from utils import qs_to_ps  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description="Sample events from a trained q-space flow-matching model (no plots)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", required=True, help="model.pt or ckpts/epNNNN.pt written by train_qspace_fm.py")
    p.add_argument("--out", required=True, help="output .pt file: p-space events (n_samples, n_particles, 3)")
    p.add_argument("--n-samples", type=int, default=100000)
    p.add_argument("--n-steps", type=int, default=500, help="Euler steps of the ODE from t = 1 to 0")
    p.add_argument("--batch-size", type=int, default=100000, help="events per sampling batch (GPU memory)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weights", choices=["ema", "raw"], default="ema", help="EMA weights or raw weights")
    p.add_argument("--save-q", action="store_true", help="also write the q-space points to <out stem>_q.pt")
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)

    model = QSpaceFlowMatching.load(a.ckpt, device=a.device, weights=a.weights)
    print(f"loaded {a.ckpt}: N = {model.cfg.n_particles}, {a.weights} weights", flush=True)
    t0 = time.time()
    torch.manual_seed(a.seed)
    Q = torch.cat([model.sample(min(a.batch_size, a.n_samples - i), n_steps=a.n_steps).cpu() for i in range(0, a.n_samples, a.batch_size)])
    P = qs_to_ps(Q)
    ok = torch.isfinite(P).all(dim=(1, 2)) & torch.isfinite(Q).all(dim=(1, 2))
    if not ok.all():
        print(f"dropping {int((~ok).sum())} non-finite events", flush=True)
        Q, P = Q[ok], P[ok]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(P, a.out)
    if a.save_q:
        torch.save(Q, os.path.splitext(a.out)[0] + "_q.pt")
    print(f"sampled {len(P)} events in {time.time() - t0:.1f}s (wall time, model already loaded) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
