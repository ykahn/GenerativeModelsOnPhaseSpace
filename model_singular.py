"""q-space phase-space diffusion model for SINGULAR distributions (qqg N=3, APS N=10).  
Self-contained: this file + train_singular.py + utils.py
are all that is needed to train it (generate.py for evaluation).

The smooth muon-decay distribution uses a different, simpler model, model_muon.py,
which is independent of this file.
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


# ---------------------------------------------------------------------------
# Configuration (defaults = the locked configuration)
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # data
    n_particles: int               # no default: always taken from the training data (train.py) or the checkpoint (load)
    # network                     
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    time_embed_dim: int = 64       
    # schedule                     (geometric small-step phase + longer/larger linear phase)
    t_steps: int = 1137            
    gamma_min: float = 0.002       
    gamma_max: float = 0.02        
    t_geom: int = 137              # number of geometric steps at the start of the FORWARD process
    gamma_geom: float = 5e-8       # first geometric step:  sigma_1 = sqrt(2 gamma_geom) = 3.2e-4
    gamma_geom_growth: float = 1.08  # gamma_k = gamma_geom * growth^k; after 137 steps, matches onto beginning of linear phase
    ref_eps: float = 0.2           # regularization of the reference score 
    # training
    batch_size: int = 1024         
    n_epochs: int = 700            
    lr: float = 3e-4               
    weight_decay: float = 1e-4     
    grad_clip: float = 1.0         
    ema_decay: float = 0.999       # EMA of the weights
    cache_dense_until: int = 150   # forward cache stores every step t <= 150 ...
    cache_stride: int = 25         # ... and every 25th step after (original: every step)
    device_str: str = "cuda"

    @property
    def input_dim(self):
        return self.n_particles * 3


# ---------------------------------------------------------------------------
# Network
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
    """pre-LN transformer block over particle tokens.
    The time embedding is added to every token before the attention layer norm."""

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
        att = att.softmax(dim=-1)        # explicit attention on purpose: the ISM loss needs a double backward through the network for BOTH
                                         # divergence estimators (Hutchinson differentiates the vector-Jacobian product v.J.v); the fused
                                         # kernels behind nn.MultiheadAttention / F.scaled_dot_product_attention have no double backward
                                         # (torch 2.7.1: "derivative for aten::_scaled_dot_product_efficient_attention_backward is not
                                         # implemented"); forcing the MATH backend works but is just this code with a fragile switch.
        o = torch.einsum("bhij,bjhd->bihd", att, v).reshape(B, n, d)
        h = h + self.proj(o)
        h = h + self.mlp(self.ln2(h))
        return h


class ScoreNetwork(nn.Module):
    """Score network s_theta(Q, t) = net(Q, t) / sigma_t.
      * one token per particle with features (q_I, q_I/|q_I|, log|q_I|) and the time embedding
        ("log-polar" features make the direction of a soft particle visible to the network);
      * n_layers transformer blocks, LayerNorm, linear head to 3 components per token;
      * the output is divided by sigma_t (the table sigma_t^2 = 2 sum_{s<t} gamma_s is a buffer, linearly
        interpolated in t), so the network itself predicts the O(1) quantity sigma_t s_theta.
    """

    def __init__(self, cfg: Config, sigmas: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.n_particles = cfg.n_particles
        self.time_embed = SinusoidalTimeEmbedding(cfg.time_embed_dim)
        self.register_buffer("sigmas", sigmas.clone())      # sigma table indexed by integer step 0..T (sigma_0 = 0)
        self.tok_dim = 3 + 4                                  # q (3) + q-hat (3) + log|q| (1)
        self.tok_in = nn.Linear(self.tok_dim + cfg.time_embed_dim, cfg.d_model)
        self.blocks = nn.ModuleList([TFBlock(cfg.d_model, cfg.n_heads, cfg.time_embed_dim) for _ in range(cfg.n_layers)])
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.out = nn.Linear(cfg.d_model, 3)
        nn.init.zeros_(self.out.weight)  
        nn.init.zeros_(self.out.bias)

    def sigma_of(self, t):
        """sigma_t for normalized t in (0, 1] by linear interpolation of the table."""
        T = self.sigmas.shape[0] - 1
        idx = t * T
        i0 = idx.floor().clamp(0, T - 1).long()
        frac = (idx - i0.to(idx.dtype)).clamp(0, 1)
        return self.sigmas[i0] * (1 - frac) + self.sigmas[i0 + 1] * frac

    def forward(self, Q, t):
        """Q: (B, N, 3) q-space point, t: (B,) normalised time in (0, 1].  Returns the score (B, N, 3)."""
        B = Q.shape[0]
        temb = self.time_embed(t)
        qn = torch.linalg.norm(Q, dim=-1, keepdim=True)
        tok = torch.cat([Q, Q / qn.clamp(min=1e-8), torch.log(qn + 1e-6),
                         temb[:, None, :].expand(B, self.n_particles, -1)], dim=-1)
        h = self.tok_in(tok)
        for blk in self.blocks:
            h = blk(h, temb)
        out = self.out(self.out_norm(h)).reshape(B, -1)
        out = out / self.sigma_of(t)[:, None]                 #sigma_t-parametrised score
        return out.reshape(B, self.n_particles, 3)


# ---------------------------------------------------------------------------
# Diffusion model
# ---------------------------------------------------------------------------
class DiffusionModel:
    TIME_WEIGHT_POWER = 2.0         #training times drawn with probability ~ (1 - t/T + 0.01)^2

    def __init__(self, cfg: Config, seed: int = -1, gammas=None):
        """gammas: explicit step-size array to use instead of the schedule built from cfg (load() passes the
        array stored in the checkpoint, so a trained model always runs with the schedule it was trained on)."""
        self.cfg = cfg
        self.device = torch.device(cfg.device_str)
        self.dtype = torch.float32
        self.seed = seed
        if seed >= 0:
            torch.manual_seed(seed)                           # seed before weight init
        self.gammas = self._build_gamma_schedule() if gammas is None else torch.as_tensor(gammas, device=self.device, dtype=self.dtype).clone()
        cum = torch.cat([torch.zeros(1, device=self.device), torch.cumsum(self.gammas, 0)])
        self.sigmas = torch.sqrt(2 * cum)                     # sigma_t^2 = 2 sum_{s<t} gamma_s, t = 0..T
        self.net = ScoreNetwork(cfg, self.sigmas).to(self.device)
        self.ema_net = None

    # -- schedule -----------------------------------------------------------
    def _build_gamma_schedule(self):
        """The first t_geom steps are a geometric sequence
        gamma_k = gamma_geom * growth^k (the original's t_gaus/gamma_gaus phase had constant steps and an OU drift);
        the remaining t_steps - t_geom steps are the original linear ramp gamma_min -> gamma_max."""
        cfg = self.cfg
        linear = torch.linspace(cfg.gamma_min, cfg.gamma_max, cfg.t_steps - cfg.t_geom, device=self.device)
        k = torch.arange(cfg.t_geom, device=self.device, dtype=torch.float32)
        geometric = cfg.gamma_geom * cfg.gamma_geom_growth ** k
        return torch.cat([geometric, linear])

    def ref_score(self, Q):
        """Regularized reference score -q^ - q/(|q|^2 + eps^2).  Used as the drift of the forward process AND in
        the reverse step, in every time step.  At q = 0 exactly (e.g. zero-padded particles) the unit vector
        qhat is undefined; we take qhat = 0, the minimal-norm element of the subdifferential of |q| (the drift on a
        null set does not affect the SDE).  The denominator is made safe before the division so that a backward
        pass through this function is finite at q = 0 as well (torch.where alone leaves 0 * nan in the gradient)."""
        qn = torch.linalg.norm(Q, dim=-1, keepdim=True)
        nonzero = qn > 0
        qhat = torch.where(nonzero, Q / torch.where(nonzero, qn, torch.ones_like(qn)), torch.zeros_like(Q))
        return -qhat - Q / (qn ** 2 + self.cfg.ref_eps ** 2)

    # -- forward process ----------------------------------------------------
    def cached_times(self):
        """integer times stored in the forward cache."""
        T, cfg = len(self.gammas), self.cfg
        return [t for t in range(1, T + 1)
                if t <= cfg.cache_dense_until or (t - cfg.cache_dense_until) % cfg.cache_stride == 0 or t == T]

    @torch.no_grad()
    def forward_process(self, q, gammas):
        """Apply the forward process to a batch of q-space vectors q (B, N, 3) with the array of step sizes
        `gammas` (any sequence; normally metadata['gammas'] = the training schedule, or a prefix of it to
        stop at an intermediate time).  Returns the final state Q_t, t = len(gammas)."""
        gammas = torch.as_tensor(gammas, device=self.device, dtype=self.dtype)
        Q = q.to(self.device, self.dtype).clone()
        for g in gammas:
            Q = Q + g * self.ref_score(Q) + torch.sqrt(2 * g) * torch.randn_like(Q)
        return Q

    @torch.no_grad()
    def precompute_cache(self, Q0, verbose=True):
        """Forward Langevin process Q_{t+1} = Q_t + gamma_t f(Q_t) + sqrt(2 gamma_t) Z 
        with the regularized drift, stored as cache[i] = Q_t for t = cached_times()[i]."""
        T = len(self.gammas)
        times = self.cached_times()
        pos = {t: i for i, t in enumerate(times)}
        cache = torch.empty((len(times), Q0.shape[0], self.cfg.n_particles, 3), device=self.device, dtype=self.dtype)
        Q = Q0.to(self.device).clone()
        t0 = time.time()
        for s in range(T):
            g = self.gammas[s]
            Q = Q + g * self.ref_score(Q) + torch.sqrt(2 * g) * torch.randn_like(Q)
            if (s + 1) in pos:
                cache[pos[s + 1]] = Q
        self._cache_times = torch.tensor(times, device=self.device)
        if verbose:
            print(f"Forward cache: {tuple(cache.shape)} ({cache.numel() * 4 / 1e9:.1f} GB) in {time.time() - t0:.1f}s", flush=True)
        return cache

    # -- ISM loss -----------------------------------------------------------
    def _score_and_div(self, Q, t):
        """Score and its divergence tr(ds/dQ).
        Hutchinson estimator tr(J) = E_v[v^T J v] with ONE Rademacher probe v, i.e. one vector-Jacobian
        product; unbiased, ~3N/2 times cheaper than exact divergence."""
        B = Q.shape[0]
        Qf = Q.reshape(B, -1).detach().requires_grad_(True)
        s = self.net(Qf.reshape(B, self.cfg.n_particles, 3), t).reshape(B, -1)
        v = torch.randint(0, 2, s.shape, device=Q.device, dtype=Q.dtype) * 2 - 1
        g = torch.autograd.grad((s * v).sum(), Qf, create_graph=True, retain_graph=True)[0]
        return s, (g * v).sum(dim=1)

    def _draw_times(self, n):
        """Cache rows and integer steps for n samples, drawn with probability ~ (1 - t/T + 0.01)^power
        over the cached times. One draw per sample, not per batch)."""
        times = self._cache_times
        if not hasattr(self, "_tw"):
            w = (1 - times.to(self.dtype) / self.cfg.t_steps + 0.01) ** self.TIME_WEIGHT_POWER
            self._tw = w / w.sum()
        rows = torch.multinomial(self._tw, n, replacement=True)
        return rows, times[rows]

    def compute_loss(self, cache):
        """Implicit score matching  E[ sigma_t^2 (0.5 |s|^2 + div s) ]  over a batch of (event, time) pairs.
        Independent t per sample and loss weight sigma_t^2."""
        cfg = self.cfg
        n_idx = torch.randint(cache.shape[1], (cfg.batch_size,), device=self.device)
        rows, t_idx = self._draw_times(cfg.batch_size)
        Q_t = cache[rows, n_idx]
        t_norm = t_idx.to(self.dtype) / cfg.t_steps
        s, div = self._score_and_div(Q_t, t_norm)
        per = 0.5 * (s * s).sum(dim=1) + div
        return (per * self.sigmas[t_idx] ** 2).mean()

    # -- training -----------------------------------------------------------
    def train(self, q_train, seed=-1, callback=None, ckpt_path=None):
        """Training loop.  AdamW + cosine annealing, grad clipping,
        loss averaged per epoch, callback(epoch, avg_loss).  EMA of the weights (decay ema_decay),
        updated after every optimizer step, is what save() stores as 'ema_state_dict' and what generate.py
        uses; the raw weights are stored as 'state_dict'.
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
        Everything is a function of the schedule `gammas`: the posterior variance replaces the 2 gamma of the
        standard Langevin reverse process, and because sigma_0 = 0, the last step
        (t_idx = 1) is automatically noiseless. Uses regularized drift.
        The signature takes (t_idx, gammas) instead of (t_normalized, gamma)."""
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
        """Apply the reverse process to a batch of q-space vectors q (B, N, 3) that sit at integer time t
        (i.e. after t forward steps), down to t = 0, with the score network `score_net` (a callable
        score_net(Q, t_normalized), e.g. model.net) and the step sizes `gammas` (the training schedule,
        metadata['gammas']).  Returns Q_0.
        """
        if seed >= 0:
            torch.manual_seed(seed)
        gammas = torch.as_tensor(gammas, device=self.device, dtype=self.dtype)
        T = len(gammas)
        Q = q.to(self.device, self.dtype).clone()
        steps = range(T - t, T)
        if verbose:
            from tqdm import tqdm
            steps = tqdm(steps, desc="Sampling")
        for s in steps:                                       # start the reverse process from forward time t
            Q = self.reverse_step(Q, T - s, gammas, score_net)
        return Q

    @torch.no_grad()
    def sample(self, n_samples, seed=-1):
        """Reverse process from the RAMBO prior p_ref over all T steps (= sample_fromQ_at_t from t = T),
            Q_{t-1} = Q_t - gamma f(Q_t) + 2 gamma s_theta(Q_t, t) + sqrt(var_t) Z .
        """
        if seed >= 0:
            torch.manual_seed(seed)
        self.net.eval()
        Q = sample_qspace(n_samples, self.cfg.n_particles, seed=seed, device=self.device, dtype=self.dtype)
        return self.sample_fromQ_at_t(self.net, Q, len(self.gammas), self.gammas)

    # -- checkpointing ------------------------------------------------------
    def save(self, path):
        """Saves model with EMA weights and the explicit schedule.
        'gammas' (the T step sizes) makes the checkpoint self-contained: load() uses this array rather than
        rebuilding the schedule from the config, so a later change of _build_gamma_schedule cannot silently
        alter the process a trained model is sampled with."""
        torch.save({"model": self.MODEL_TAG, "config": asdict(self.cfg), "seed": self.seed,
                    "gammas": self.gammas.detach().cpu().clone(),
                    "state_dict": self.net.state_dict(),
                    "ema_state_dict": (self.ema_net.state_dict() if self.ema_net is not None else None)}, path)

    MODEL_TAG = "singular"          # written into checkpoints; sample.py dispatches on it

    @classmethod
    def load(cls, path, device="cuda", weights="ema"):
        """Load a checkpoint written by save().  weights = 'ema' (the model of record) or 'raw'.
        The schedule is the 'gammas' array stored in the checkpoint (never rebuilt from the config); it must be
        consistent with the sigma table stored in the network weights, which is what the network was trained with."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("model", "singular") != cls.MODEL_TAG:      # (no tag = written by a version of this code before the muon model existed)
            raise ValueError(f"{path} is a '{ck['model']}' checkpoint, not '{cls.MODEL_TAG}'")
        known = {f.name for f in fields(Config)}
        cfg = Config(**{k: v for k, v in ck["config"].items() if k in known})   # keys of older versions of this code are ignored
        cfg.device_str = device
        sd = ck["ema_state_dict"] if weights == "ema" and ck.get("ema_state_dict") is not None else ck["state_dict"]
        m = cls(cfg, gammas=ck["gammas"])
        if not torch.allclose(m._build_gamma_schedule(), m.gammas, rtol=1e-6, atol=0):
            print(f"WARNING {path}: the schedule stored in the checkpoint differs from the one _build_gamma_schedule builds "
                  f"from its config; using the stored schedule.", flush=True)
        if "sigmas" in sd and not torch.allclose(sd["sigmas"].to(m.sigmas), m.sigmas, rtol=1e-5, atol=0):
            raise ValueError(f"{path}: schedule inconsistent with the sigma table the network was trained with")
        m.net.load_state_dict(sd)
        m.net.eval()
        return m


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
APS_B = (0.0, 0.0, 0.0) #a nonzero boost only adds anisotropy and doesn't help anything
APS_X = 0.0846 #this is the value which works best for the N = 10 APS events; use x = 0.15 for the N = 3 events

def load_pspace(path, n=0):
    """p-space events (N_events, N, 3) from a .pt file; n > 0 keeps the first n events (original: n_train)."""
    p = torch.load(path, map_location="cpu", weights_only=True).float()
    return p[:n] if n > 0 and p.shape[0] > n else p


def embed_fixed(ps, b=APS_B, x=APS_X):
    """Single copy with one fixed boost b and scale x,  q = Lambda(-b) p / x  (utils.ps_to_qs)."""
    return ps_to_qs(ps, torch.tensor([b], dtype=ps.dtype), torch.tensor([x], dtype=ps.dtype))
