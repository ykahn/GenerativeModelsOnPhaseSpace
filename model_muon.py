"""Score model of record for the MUON-DECAY distribution (smooth, labelled 3-body final state).  Self-contained:
this file + train_muon.py + utils.py are all that is needed to train it (sample.py / evolution.py evaluate it).

This configuration reproduces and slightly improves on the paper's muon-decay results (arXiv:2604.02415 Figs. 1-5,
ablation of Sept 2026).  It is deliberately the ORIGINAL paper model with only the generic training/sampling
improvements kept.  Reference for every "UNCHANGED"/"CHANGE" below: phasespace_diffusion/model.py and train.py of
the original package.

  network      UNCHANGED  the paper's MLP: 4 layers of width 256, SiLU, on the flat 3N-vector + 64-dim sinusoidal
                          embedding of t/T, zero-initialised output layer; output IS the score (no sigma_t scaling).
                          Not permutation-equivariant, so it can represent labelled particles (the transformer of
                          model_singular.py cannot: it learns the permutation-symmetrised density).
  drift        UNCHANGED  the paper's reference score -(1 + 1/|q|) q^  (no regularisation).
  schedule     UNCHANGED  the paper's linear gamma 0.002 -> 0.01 over T = 500 steps (total time 3), no geometric phase.
  loss         UNCHANGED  implicit score matching with the EXACT divergence (3N backward passes), loss weight
                          (1 - t/T + 0.01), ONE time step per batch drawn with probability ~ (1 - t/T + 0.01)^2,
                          forward process cached at every step.
  embedding    CHANGE     one fixed copy with b = 0 and x = 0.15 (the data then sits at the prior's scale, mean |q| ~ 2;
                          the paper's run used a random RAMBO (b, x) with x = 0.18).  b = 0 removes the boost-induced
                          angular anisotropy.
  optimiser    CHANGE     AdamW lr 3e-4 (paper: 1e-3), 500 epochs on all 500k events (paper: 100 epochs), EMA 0.999
                          of the weights is the model (paper: lowest-loss epoch); periodic checkpoints.
  sampler      CHANGE     posterior-variance noise 2 gamma sigma_{t-1}^2/sigma_t^2 and a noiseless last step
                          (paper: Euler-Maruyama with noise 2 gamma).

Why not the singular model of model_singular.py: on this distribution its sigma_t^2 loss weighting over-populates
the soft tail by a factor 2-3 and its geometric small-step phase slows the bulk fit badly (ablation of Sept 2026,
improved/STATE.md).
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
from utils import ps_to_qs, sample_qspace  # noqa: E402  (verbatim copies of the original helpers)


# ---------------------------------------------------------------------------
# Configuration (defaults = the muon model of record)
# ---------------------------------------------------------------------------
@dataclass
class MuonConfig:
    n_particles: int = 3
    # network (paper)
    hidden_dim: int = 256
    n_layers: int = 4
    time_embed_dim: int = 64
    # schedule (paper): linear only
    t_steps: int = 500
    gamma_min: float = 0.002
    gamma_max: float = 0.01
    # training
    batch_size: int = 1024         # paper
    n_epochs: int = 500            # paper: 100
    lr: float = 3e-4               # paper: 1e-3
    weight_decay: float = 1e-4     # paper
    grad_clip: float = 1.0         # paper
    ema_decay: float = 0.999       # CHANGE (paper: lowest-loss epoch)
    device_str: str = "cuda"

    @property
    def input_dim(self):
        return self.n_particles * 3


MUON_B = (0.0, 0.0, 0.0)   # no boost: a nonzero b only adds anisotropy
MUON_X = 0.15              # puts the 3-body data at the prior's scale (mean |q| ~ 2)


# ---------------------------------------------------------------------------
# Network (UNCHANGED from the original ScoreNetwork)
# ---------------------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    """UNCHANGED from the original."""

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


class MuonScoreNetwork(nn.Module):
    """UNCHANGED architecture of the original ScoreNetwork: n_layers x (Linear, SiLU) on [flat Q, time embedding],
    linear output to 3N, zero-initialised.  The output is the score itself."""

    def __init__(self, cfg: MuonConfig):
        super().__init__()
        self.cfg = cfg
        self.n_particles = cfg.n_particles
        self.time_embed = SinusoidalTimeEmbedding(cfg.time_embed_dim)
        layers, d = [], 3 * cfg.n_particles + cfg.time_embed_dim
        for _ in range(cfg.n_layers):
            layers += [nn.Linear(d, cfg.hidden_dim), nn.SiLU()]
            d = cfg.hidden_dim
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(cfg.hidden_dim, 3 * cfg.n_particles)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, Q, t):
        """Q: (B, N, 3) q-space point, t: (B,) normalised time in (0, 1].  Returns the score (B, N, 3)."""
        B = Q.shape[0]
        out = self.out(self.net(torch.cat([Q.reshape(B, -1), self.time_embed(t)], dim=-1)))
        return out.reshape(B, self.n_particles, 3)


# ---------------------------------------------------------------------------
# Diffusion model
# ---------------------------------------------------------------------------
class MuonDiffusionModel:
    MODEL_TAG = "muon"              # written into checkpoints; sample.py dispatches on it
    TIME_WEIGHT_POWER = 2.0         # paper: one t per batch drawn with probability ~ (1 - t/T + 0.01)^2
    LOSS_WEIGHT_POWER = 1.0         # paper: loss weight (1 - t/T + 0.01)^1

    def __init__(self, cfg: MuonConfig, seed: int = -1, gammas=None):
        """gammas: explicit step-size array to use instead of the schedule built from cfg (load() passes the
        array stored in the checkpoint, so a trained model always runs with the schedule it was trained on)."""
        self.cfg = cfg
        self.device = torch.device(cfg.device_str)
        self.dtype = torch.float32
        self.seed = seed
        if seed >= 0:
            torch.manual_seed(seed)                           # same as original: seed before weight init
        self.gammas = self._build_gamma_schedule() if gammas is None else torch.as_tensor(gammas, device=self.device, dtype=self.dtype).clone()
        cum = torch.cat([torch.zeros(1, device=self.device), torch.cumsum(self.gammas, 0)])
        self.sigmas = torch.sqrt(2 * cum)                     # sigma_t^2 = 2 sum_{s<t} gamma_s, t = 0..T (sampler only)
        self.net = MuonScoreNetwork(cfg).to(self.device)
        self.ema_net = None

    # -- schedule: the paper's linear ramp ----------------------------------
    def _build_gamma_schedule(self):
        """UNCHANGED: linear gamma_min -> gamma_max over t_steps."""
        cfg = self.cfg
        return torch.linspace(cfg.gamma_min, cfg.gamma_max, cfg.t_steps, device=self.device)

    # -- drift: the paper's reference score ----------------------------------
    def ref_score(self, Q):
        """UNCHANGED: the paper's reference score -(1 + 1/|q|) q^ = -(1 + 1/|q|)/|q| q  (qspace_score of the original package)."""
        q = torch.linalg.norm(Q, dim=2)
        return -(1 + 1 / q[:, :, None]) / q[:, :, None] * Q

    # -- forward process ----------------------------------------------------
    @torch.no_grad()
    def forward_process(self, q, gammas):
        """Apply the forward process to a batch of q-space vectors q (B, N, 3) with the array of step sizes
        `gammas` (normally metadata['gammas'] = the training schedule, or a prefix of it to stop at an
        intermediate time).  Returns the final state Q_t, t = len(gammas).  UNCHANGED from the original
        forward_process (without its OU option)."""
        gammas = torch.as_tensor(gammas, device=self.device, dtype=self.dtype)
        Q = q.to(self.device, self.dtype).clone()
        for g in gammas:
            Q = Q + g * self.ref_score(Q) + torch.sqrt(2 * g) * torch.randn_like(Q)
        return Q

    @torch.no_grad()
    def precompute_cache(self, Q0, verbose=True):
        """UNCHANGED: the forward Langevin process Q_{t+1} = Q_t + gamma_t f(Q_t) + sqrt(2 gamma_t) Z, cached at every step:
        cache[t - 1] = Q_t for t = 1..T."""
        T = len(self.gammas)
        cache = torch.empty((T, Q0.shape[0], self.cfg.n_particles, 3), device=self.device, dtype=self.dtype)
        Q = Q0.to(self.device).clone()
        t0 = time.time()
        for s in range(T):
            g = self.gammas[s]
            Q = Q + g * self.ref_score(Q) + torch.sqrt(2 * g) * torch.randn_like(Q)
            cache[s] = Q
        self._cache_times = torch.arange(1, T + 1, device=self.device)
        if verbose:
            print(f"Forward cache: {tuple(cache.shape)} ({cache.numel() * 4 / 1e9:.1f} GB) in {time.time() - t0:.1f}s", flush=True)
        return cache

    # -- ISM loss (UNCHANGED) -------------------------------------------------
    def _score_and_div(self, Q, t):
        """UNCHANGED: exact divergence, one backward pass per input component (3N = 9 passes)."""
        B = Q.shape[0]
        Qf = Q.reshape(B, -1).detach().requires_grad_(True)
        s = self.net(Qf.reshape(B, self.cfg.n_particles, 3), t).reshape(B, -1)
        div = torch.zeros(B, device=Q.device, dtype=Q.dtype)
        for i in range(self.cfg.input_dim):
            g = torch.autograd.grad(s[:, i].sum(), Qf, create_graph=True, retain_graph=True)[0]
            div = div + g[:, i]
        return s, div

    def _draw_time(self):
        """One cache row / integer step per batch, drawn with probability ~ (1 - t/T + 0.01)^2 (UNCHANGED)."""
        times = self._cache_times
        if not hasattr(self, "_tw"):
            w = (1 - times.to(self.dtype) / self.cfg.t_steps + 0.01) ** self.TIME_WEIGHT_POWER
            self._tw = w / w.sum()
        row = torch.multinomial(self._tw, 1, replacement=True)
        return row, times[row]

    def compute_loss(self, cache):
        """UNCHANGED objective of the original _compute_ism_loss_cached:  (1 - t/T + 0.01) * mean(0.5 |s|^2 + div s)
        with ONE time step per batch."""
        cfg = self.cfg
        n_idx = torch.randint(cache.shape[1], (cfg.batch_size,), device=self.device)
        row, t_idx = self._draw_time()
        Q_t = cache[row.expand(cfg.batch_size), n_idx]
        t_norm = t_idx.expand(cfg.batch_size).to(self.dtype) / cfg.t_steps
        s, div = self._score_and_div(Q_t, t_norm)
        per = 0.5 * (s * s).sum(dim=1) + div
        w = (1 - t_norm + 0.01) ** self.LOSS_WEIGHT_POWER
        return (per * w).mean()

    # -- training -----------------------------------------------------------
    def train(self, q_train, seed=-1, callback=None, ckpt_path=None):
        """Training loop.  Same structure as the original train(): AdamW + cosine annealing, grad clipping, loss
        averaged per epoch, callback(epoch, avg_loss).  CHANGE: an EMA of the weights (decay ema_decay), updated
        after every optimiser step, is what save() stores as 'ema_state_dict' and what sample.py uses; the raw
        weights are stored as 'state_dict'.  The original kept the lowest-loss epoch instead.
        model.pt is (re)written every 10 epochs so an interrupted job leaves a usable model."""
        cfg = self.cfg
        if seed >= 0:
            torch.manual_seed(seed)
        cache = self.precompute_cache(q_train)
        n_batches = q_train.shape[0] // cfg.batch_size
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
                opt.zero_grad(set_to_none=True)
                loss = self.compute_loss(cache)
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
        del cache
        torch.cuda.empty_cache()
        return losses

    # -- sampling -----------------------------------------------------------
    @torch.no_grad()
    def reverse_step(self, Q, t_idx, gammas, score_net):
        """One reverse Euler step from integer time t_idx to t_idx - 1:
            Q - gamma_t f(Q) + 2 gamma_t s_theta(Q, t/T) + sqrt(var_t) Z ,
            var_t = 2 gamma_t sigma_{t-1}^2 / sigma_t^2 ,   sigma_t^2 = 2 sum_{s<t} gamma_s ,
        where gamma_t = gammas[t_idx - 1] is the forward step that led from t_idx - 1 to t_idx.
        CHANGE vs the original reverse_step: the noise variance is the posterior variance (original: 2 gamma), and
        because sigma_0 = 0 the last step (t_idx = 1) is automatically noiseless.  Drift terms unchanged."""
        gammas = torch.as_tensor(gammas, device=self.device, dtype=self.dtype)
        T = len(gammas)
        gamma = gammas[t_idx - 1]
        cum = torch.cumsum(gammas, 0)                                   # cum[t-1] = sum_{s<t} gamma_s
        sig2_t = torch.sqrt(2 * cum[t_idx - 1]) ** 2                    # (via sigma_t, as in __init__)
        sig2_prev = torch.sqrt(2 * cum[t_idx - 2]) ** 2 if t_idx > 1 else torch.zeros((), device=self.device, dtype=self.dtype)
        var = 2 * gamma * sig2_prev / sig2_t
        t_normalized = torch.full((Q.shape[0],), t_idx / T, device=self.device, dtype=self.dtype)
        s_model = score_net(Q, t_normalized)
        s_ref = self.ref_score(Q)
        if t_idx > 1:
            return Q - gamma * s_ref + 2 * gamma * s_model + torch.sqrt(var) * torch.randn_like(Q)
        return Q - gamma * s_ref + 2 * gamma * s_model                  # var = 0: no noise in the last step

    @torch.no_grad()
    def sample_fromQ_at_t(self, score_net, q, t, gammas, seed=-1, verbose=False):
        """Apply the reverse process to a batch of q-space vectors q (B, N, 3) that sit at integer time t (i.e. after
        t forward steps), down to t = 0, with the score network `score_net` (a callable score_net(Q, t_normalized),
        e.g. model.net) and the step sizes `gammas` (the training schedule, metadata['gammas']).  Returns Q_0.
        Same loop as sample_fromQ_at_t of omnilearn_lightning/diffusion.py: s = T - t .. T - 1, integer times t .. 1."""
        if seed >= 0:
            torch.manual_seed(seed)
        gammas = torch.as_tensor(gammas, device=self.device, dtype=self.dtype)
        T = len(gammas)
        Q = q.to(self.device, self.dtype).clone()
        steps = range(T - t, T)
        if verbose:
            from tqdm import tqdm
            steps = tqdm(steps, desc="Sampling")
        for s in steps:
            Q = self.reverse_step(Q, T - s, gammas, score_net)
        return Q

    @torch.no_grad()
    def sample(self, n_samples, seed=-1):
        """Reverse process from the RAMBO prior p_ref over all T steps (= sample_fromQ_at_t from t = T)."""
        if seed >= 0:
            torch.manual_seed(seed)
        self.net.eval()
        Q = sample_qspace(n_samples, self.cfg.n_particles, seed=seed, device=self.device, dtype=self.dtype)
        return self.sample_fromQ_at_t(self.net, Q, len(self.gammas), self.gammas)

    # -- checkpointing ------------------------------------------------------
    def save(self, path):
        """Same format as the original save() plus the EMA weights, the explicit schedule and the model tag."""
        torch.save({"model": self.MODEL_TAG, "config": asdict(self.cfg), "seed": self.seed,
                    "gammas": self.gammas.detach().cpu().clone(),
                    "state_dict": self.net.state_dict(),
                    "ema_state_dict": (self.ema_net.state_dict() if self.ema_net is not None else None)}, path)

    @classmethod
    def load(cls, path, device="cuda", weights="ema"):
        """Load a checkpoint written by save().  weights = 'ema' (the model of record) or 'raw'.
        The schedule is the 'gammas' array stored in the checkpoint (never rebuilt from the config)."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("model") != cls.MODEL_TAG:
            raise ValueError(f"{path} is a '{ck.get('model')}' checkpoint, not '{cls.MODEL_TAG}'")
        known = {f.name for f in fields(MuonConfig)}
        cfg = MuonConfig(**{k: v for k, v in ck["config"].items() if k in known})   # keys of older versions of this code are ignored
        cfg.device_str = device
        sd = ck["ema_state_dict"] if weights == "ema" and ck.get("ema_state_dict") is not None else ck["state_dict"]
        m = cls(cfg, gammas=ck["gammas"])
        if not torch.allclose(m._build_gamma_schedule(), m.gammas, rtol=1e-6, atol=0):
            print(f"WARNING {path}: the schedule stored in the checkpoint differs from the one _build_gamma_schedule builds "
                  f"from its config; using the stored schedule.", flush=True)
        m.net.load_state_dict(sd)
        m.net.eval()
        return m


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_pspace(path, n=0):
    """p-space events (N_events, N, 3) from a .pt file; n > 0 keeps the first n events."""
    p = torch.load(path, map_location="cpu", weights_only=True).float()
    return p[:n] if n > 0 and p.shape[0] > n else p


def embed_fixed(ps, b=MUON_B, x=MUON_X):
    """CHANGE vs original fluff_in_q_space (N_mult random (b, x) copies of the data): a single copy with one
    fixed boost b and scale x,  q = Lambda(-b) p / x  (utils.ps_to_qs, unchanged)."""
    return ps_to_qs(ps, torch.tensor([b], dtype=ps.dtype), torch.tensor([x], dtype=ps.dtype))
