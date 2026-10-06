"""Sample events from a trained p-space flow-matching model (model_pspace_fm.py).  Sampling only: no plots, no comparison with the truth.

Usage:
    python generate_pspace_fm.py --ckpt runs/pspace_fm_xi30/model.pt --n-samples 100000 --out runs/pspace_fm_xi30/generated.pt
                                 [--n-steps 500] [--weights ema|raw] [--seed 0] [--batch-size 100000] [--device cuda]

Integrates the learned ODE with --n-steps Euler steps from the prior N(0, I) (t = 1) to t = 0 and writes the
p-space 3-momenta (n_samples, n_particles, 3), float32, to --out, after undoing the 1/x
rescaling of the training data (x from the checkpoint).  The events do not conserve energy-momentum; the mean
violations in units of the median particle energy of each event (paper Table I) are printed.
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_pspace_fm import PSpaceFlowMatching  # noqa: E402


def conservation_violation(P):
    """mean |sum_I E_I - 1| / E~ and mean |sum_I p_I,k| / E~ (k = x, y, z), E~ = median particle energy of the event."""
    E = P.norm(dim=-1)
    Emed = E.median(dim=1).values
    return [((E.sum(1) - 1).abs() / Emed).mean().item()] + (P.sum(1).abs() / Emed[:, None]).mean(0).tolist()


def main(argv=None):
    p = argparse.ArgumentParser(description="Sample events from a trained p-space flow-matching model (no plots)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", required=True, help="model.pt or ckpts/epNNNN.pt written by train_pspace_fm.py")
    p.add_argument("--out", required=True, help="output .pt file: p-space events (n_samples, n_particles, 3)")
    p.add_argument("--n-samples", type=int, default=100000)
    p.add_argument("--n-steps", type=int, default=500, help="Euler steps of the ODE from t = 1 to 0")
    p.add_argument("--batch-size", type=int, default=100000, help="events per sampling batch (GPU memory)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weights", choices=["ema", "raw"], default="ema", help="EMA weights or raw weights")
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)

    model = PSpaceFlowMatching.load(a.ckpt, device=a.device, weights=a.weights)
    print(f"loaded {a.ckpt}: N = {model.cfg.n_particles}, x = {model.cfg.x}, {a.weights} weights", flush=True)
    t0 = time.time()
    torch.manual_seed(a.seed)
    P = torch.cat([model.sample(min(a.batch_size, a.n_samples - i), n_steps=a.n_steps).cpu() for i in range(0, a.n_samples, a.batch_size)]) * model.cfg.x   # undo the 1/x of training
    ok = torch.isfinite(P).all(dim=(1, 2))
    if not ok.all():
        print(f"dropping {int((~ok).sum())} non-finite events", flush=True)
        P = P[ok]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(P, a.out)
    print(f"sampled {len(P)} events in {time.time() - t0:.1f}s (wall time, model already loaded) -> {a.out}", flush=True)
    print("mean conservation violation / median E (E, px, py, pz): " + " ".join(f"{v:.4f}" for v in conservation_violation(P)), flush=True)


if __name__ == "__main__":
    main()
