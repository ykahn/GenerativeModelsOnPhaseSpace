"""Comparison model: flow matching in q-space with the RAMBO prior p_ref.
Self-contained: this file + utils.py + train_qspace_fm.py (training) + generate_qspace_fm.py (sampling).

  data        the fixed embedding of model_singular: q = Lambda(-b) p / x, b = 0, x = 0.0846 (APS N=10).
  path        Eq. (12):  Q_t = (1 - t) Q + t Q~,  t ~ U(0, 1),  Q ~ data, Q~ ~ p_ref (RAMBO q-space, utils.sample_qspace),
              drawn independently (no coupling).
  loss        Eq. (13):  |v_theta(Q_t, t) - (Q~ - Q)|^2.
  sampler     Q~ ~ p_ref, Euler integration of dQ/dt = v_theta(Q, t) from t = 1 to t = 0 (n_steps uniform steps), then
              p = qs_to_ps(Q) (energy-momentum conserved exactly).
  network     the transformer of model_singular.ScoreNetwork, unchanged (same blocks, token features (q_I, q^_I, log|q_I|),
              64-dim time embedding of t, zero-initialised output layer, same parameter names), except that the output is
              NOT divided by sigma_t: it is the velocity v_theta(Q, t).
  optimiser   as model_singular: AdamW 3e-4, weight decay 1e-4, clip 1, cosine, batch 1024, 700 epochs, EMA 0.999.
"""

import copy
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import ps_to_qs, sample_qspace  # noqa: E402


@dataclass
class Config:
    n_particles: int               # no default: always taken from the training data or the checkpoint
    # network (= model_singular.Config)
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    time_embed_dim: int = 64
    # training (= model_singular.Config)
    batch_size: int = 1024
    n_epochs: int = 700
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    device_str: str = "cuda"


# ---------------------------------------------------------------------------
# Network (verbatim from model_singular.py except for the output scaling)
# ---------------------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period)
                          * torch.arange(half, device=t.device, dtype=t.dtype) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TFBlock(nn.Module):
    """Pre-LN transformer block over particle tokens; the time embedding is added to every token before the attention
    layer norm.  Explicit attention kept as in model_singular (no double backward is needed here, but this keeps the
    network numerically identical)."""

    def __init__(self, d, n_heads, temb_dim):
        super().__init__()
        self.d, self.nh = d, n_heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))
        self.temb = nn.Linear(temb_dim, d)

    def forward(self, h, temb):
        B, n, d = h.shape
        x = self.ln1(h + self.temb(temb)[:, None, :])
        q, k, v = self.qkv(x).reshape(B, n, 3, self.nh, d // self.nh).unbind(2)          # (B, n, H, dh)
        att = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(d // self.nh)
        att = att.softmax(dim=-1)
        o = torch.einsum("bhij,bjhd->bihd", att, v).reshape(B, n, d)
        h = h + self.proj(o)
        h = h + self.mlp(self.ln2(h))
        return h


class VelocityNetwork(nn.Module):
    """v_theta(Q, t): model_singular.ScoreNetwork without the sigma table and without the division by sigma_t."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_particles = cfg.n_particles
        self.time_embed = SinusoidalTimeEmbedding(cfg.time_embed_dim)
        self.tok_dim = 3 + 4                                  # q (3) + q^ (3) + log|q| (1)
        self.tok_in = nn.Linear(self.tok_dim + cfg.time_embed_dim, cfg.d_model)
        self.blocks = nn.ModuleList([TFBlock(cfg.d_model, cfg.n_heads, cfg.time_embed_dim) for _ in range(cfg.n_layers)])
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.out = nn.Linear(cfg.d_model, 3)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, Q, t):
        """Q: (B, N, 3), t: (B,) flow time in [0, 1].  Returns v_theta (B, N, 3)."""
        B = Q.shape[0]
        temb = self.time_embed(t)
        qn = torch.linalg.norm(Q, dim=-1, keepdim=True)
        tok = torch.cat([Q, Q / qn.clamp(min=1e-8), torch.log(qn + 1e-6),
                         temb[:, None, :].expand(B, self.n_particles, -1)], dim=-1)
        h = self.tok_in(tok)
        for blk in self.blocks:
            h = blk(h, temb)
        return self.out(self.out_norm(h))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class QSpaceFlowMatching:
    MODEL_TAG = "qspace_fm"

    def __init__(self, cfg: Config, seed: int = -1):
        self.cfg = cfg
        self.device = torch.device(cfg.device_str)
        self.dtype = torch.float32
        self.seed = seed
        if seed >= 0:
            torch.manual_seed(seed)
        self.net = VelocityNetwork(cfg).to(self.device)
        self.ema_net = None

    def prior(self, n):
        """RAMBO q-space prior p_ref (Eq. (8)), drawn from the global torch RNG state."""
        return sample_qspace(n, self.cfg.n_particles, seed=int(torch.randint(2 ** 62, ())), device=self.device, dtype=self.dtype)

    # -- loss ---------------------------------------------------------------
    def compute_loss(self, Q0):
        """Eqs. (12)-(13) on a batch of data Q0 (B, N, 3) (already on the device)."""
        Q1 = self.prior(Q0.shape[0])
        t = torch.rand(Q0.shape[0], device=self.device, dtype=self.dtype)
        Qt = (1 - t)[:, None, None] * Q0 + t[:, None, None] * Q1
        return ((self.net(Qt, t) - (Q1 - Q0)) ** 2).sum(dim=(1, 2)).mean()

    # -- training -----------------------------------------------------------
    def train(self, q_train, seed=-1, callback=None, ckpt_path=None):
        """Same loop as model_singular.DiffusionModel.train: AdamW + cosine annealing, grad clipping, EMA after every
        step, callback(epoch, avg_loss), model.pt rewritten every 10 epochs.  q_train: (N_events, N, 3)."""
        cfg = self.cfg
        if seed >= 0:
            torch.manual_seed(seed)
        data = q_train.to(self.device, self.dtype)
        n_batches = data.shape[0] // cfg.batch_size
        opt = optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_epochs)
        self.ema_net = copy.deepcopy(self.net)
        for p in self.ema_net.parameters():
            p.requires_grad_(False)
        losses = []
        print(f"Model parameters: {sum(p.numel() for p in self.net.parameters()):,}", flush=True)
        print(f"Training {cfg.n_epochs} epochs x {n_batches} batches of {cfg.batch_size}", flush=True)
        self.net.train()
        t0 = time.time()
        for epoch in range(cfg.n_epochs):
            ep = []
            for _ in range(n_batches):
                idx = torch.randint(data.shape[0], (cfg.batch_size,), device=self.device)
                opt.zero_grad(set_to_none=True)
                loss = self.compute_loss(data[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
                opt.step()
                with torch.no_grad():
                    for pe, p in zip(self.ema_net.parameters(), self.net.parameters()):
                        pe.mul_(cfg.ema_decay).add_(p.detach(), alpha=1 - cfg.ema_decay)
                ep.append(loss.item())
            sched.step()
            avg = float(np.mean(ep))
            losses.append(avg)
            if callback is not None:
                callback(epoch, avg)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{cfg.n_epochs} loss {avg:.4f} lr {sched.get_last_lr()[0]:.2e} [{time.time() - t0:.0f}s]", flush=True)
                if ckpt_path is not None:
                    self.save(ckpt_path)
        return losses

    # -- sampling -----------------------------------------------------------
    @torch.no_grad()
    def sample(self, n_samples, n_steps=500, seed=-1):
        """Q~ ~ p_ref, Euler steps of dQ/dt = v_theta from t = 1 to 0.  Returns q-space points (map with utils.qs_to_ps)."""
        if seed >= 0:
            torch.manual_seed(seed)
        self.net.eval()
        Q = self.prior(n_samples)
        dt = 1.0 / n_steps
        for k in range(n_steps):
            t = torch.full((n_samples,), 1 - k * dt, device=self.device, dtype=self.dtype)
            Q = Q - dt * self.net(Q, t)
        return Q

    # -- checkpointing ------------------------------------------------------
    def save(self, path):
        torch.save({"model": self.MODEL_TAG, "config": asdict(self.cfg), "seed": self.seed,
                    "state_dict": self.net.state_dict(),
                    "ema_state_dict": (self.ema_net.state_dict() if self.ema_net is not None else None)}, path)

    @classmethod
    def load(cls, path, device="cuda", weights="ema"):
        """weights = 'ema' (default) or 'raw'."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("model") != cls.MODEL_TAG:
            raise ValueError(f"{path} is a '{ck.get('model')}' checkpoint, not '{cls.MODEL_TAG}'")
        known = {f.name for f in fields(Config)}
        cfg = Config(**{k: v for k, v in ck["config"].items() if k in known})
        cfg.device_str = device
        m = cls(cfg)
        sd = ck["ema_state_dict"] if weights == "ema" and ck.get("ema_state_dict") is not None else ck["state_dict"]
        m.net.load_state_dict(sd)
        m.net.eval()
        return m


# ---------------------------------------------------------------------------
# Data (= model_singular.py)
# ---------------------------------------------------------------------------
APS_B = (0.0, 0.0, 0.0)
APS_X = 0.0846


def load_pspace(path, n=0):
    """p-space events (N_events, N, 3) from a .pt file; n > 0 keeps the first n events."""
    p = torch.load(path, map_location="cpu", weights_only=True).float()
    return p[:n] if n > 0 and p.shape[0] > n else p


def embed_fixed(ps, b=APS_B, x=APS_X):
    """q = Lambda(-b) p / x  (utils.ps_to_qs) with one fixed boost b and scale x."""
    return ps_to_qs(ps, torch.tensor([b], dtype=ps.dtype), torch.tensor([x], dtype=ps.dtype))
