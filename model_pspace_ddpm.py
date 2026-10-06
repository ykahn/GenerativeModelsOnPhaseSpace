"""Comparison model: standard DDPM (Ornstein-Uhlenbeck diffusion, "predict the noise") directly on the p-space
3-momenta P = {p_I}, with no q-space map and no reference score.  Self-contained: this file + train_pspace_ddpm.py
(training) + generate_pspace_ddpm.py (sampling).  Generated events do NOT conserve energy-momentum (paper Table I).

  data        P / x with x = 0.0846, the scale of the q-space embedding of model_singular (component std 0.091 -> 1.08);
              train_pspace_ddpm.py divides by x, generate_pspace_ddpm.py multiplies the samples by x (stored in the config).
  network     the transformer of model_singular.ScoreNetwork, unchanged (same blocks, token features (p_I, p^_I, log|p_I|),
              time embedding of t/T, zero-initialised output layer, same parameter names), except for the output
              parametrisation: the network output F_theta is NOT divided by sigma_t but preconditioned (Karras et al. 2022):
                  eps_theta(P_t, t) = c_skip(t) P_t + c_out(t) F_theta(P_t, t/T),
                  c_skip = sqrt(1-abar_t) / (abar_t s^2 + 1-abar_t),   c_out = sqrt(abar_t s^2 / (abar_t s^2 + 1-abar_t)),
              s = data_std (pooled component std of the training data).  c_skip P_t is the exact eps for Gaussian data of std
              s, so F_theta = 0 (the zero-initialised network) samples that Gaussian, and F_theta has an O(1) target at all t.
              Why: with plain eps_theta = F_theta the sampler is exponentially sensitive to eps errors in the over-converged
              part of this schedule (the reverse mean multiplies by 1/sqrt(1-beta_t), prod = 1/sqrt(abar_T) = 7e4, which
              eps ~ P_t must cancel; the implied P_0 error is delta/sqrt(abar_t)) and a partially trained model generated
              events with |p| ~ 1e3.  With c_out ~ sqrt(abar_t) s at large t the implied P_0 error is ~ s delta instead.
              Small t: c_skip -> 0, c_out -> 1, i.e. plain eps prediction.  The loss is unchanged.
  forward     paper Eq. (A5):  P_{t+1} = sqrt(1 - 2 gamma_t) P_t + sqrt(2 gamma_t) Z_t,  i.e. DDPM with beta_t = 2 gamma_t,
              gamma_t = the schedule of model_singular (137 geometric steps 5e-8 * 1.08^k, then linear 0.002 -> 0.02,
              T = 1137).  Closed form  P_t = sqrt(abar_t) P_0 + sqrt(1 - abar_t) eps,  abar_t = prod_{s<=t} (1 - beta_s),
              so no forward cache is needed.
              Convergence to the prior N(0, I), checked on SARGE_N10_xi30_1M (unscaled data, component std 0.091):
              sqrt(abar_T) = 1.4e-5 (residual signal ~1e-6; ~2e-5 for the rescaled data P / x); iterating the 1137 steps on 200k events gives component
              mean -4e-4, std 0.9999, KS distance to N(0,1) 2.6e-4 (an exact N(0,1) sample of the same size: 2.7e-4),
              corr(P_T, P_0) = -2e-5.  Converged by t ~ 700 already (sqrt(abar) = 0.018 there).
  loss        Ho et al. L_simple:  |eps_theta(P_t, t/T) - eps|^2,  t uniform in 1..T, independently per sample.
  sampler     ancestral DDPM from P_T ~ N(0, I):  P_{t-1} = (P_t - beta_t/sqrt(1-abar_t) eps_theta)/sqrt(1-beta_t) + sqrt(var_t) Z,
              var_t = beta_t (1-abar_{t-1})/(1-abar_t) (posterior variance, as in model_singular); the last step is noiseless.
  optimiser   as model_singular: AdamW 3e-4, weight decay 1e-4, clip 1, cosine, batch 1024, 700 epochs, EMA 0.999.
"""

import copy
import math
import time
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class Config:
    n_particles: int               # no default: always taken from the training data or the checkpoint
    x: float = 0.0846              # data scale: the model works with P / x (x = the q-space embedding scale of model_singular)
    data_std: float = 1.0          # s of the output preconditioning: pooled component std of P / x (set by train_pspace_ddpm.py)
    # network (= model_singular.Config)
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    time_embed_dim: int = 64
    # schedule (= model_singular.Config; beta_t = 2 gamma_t)
    t_steps: int = 1137
    gamma_min: float = 0.002
    gamma_max: float = 0.02
    t_geom: int = 137
    gamma_geom: float = 5e-8
    gamma_geom_growth: float = 1.08
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


class EpsNetwork(nn.Module):
    """F_theta(P, t): model_singular.ScoreNetwork without the sigma table and without the division by sigma_t
    (PSpaceDDPM.eps turns it into the noise prediction)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_particles = cfg.n_particles
        self.time_embed = SinusoidalTimeEmbedding(cfg.time_embed_dim)
        self.tok_dim = 3 + 4                                  # p (3) + p^ (3) + log|p| (1)
        self.tok_in = nn.Linear(self.tok_dim + cfg.time_embed_dim, cfg.d_model)
        self.blocks = nn.ModuleList([TFBlock(cfg.d_model, cfg.n_heads, cfg.time_embed_dim) for _ in range(cfg.n_layers)])
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.out = nn.Linear(cfg.d_model, 3)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, P, t):
        """P: (B, N, 3), t: (B,) normalised time t/T in (0, 1].  Returns F_theta (B, N, 3)."""
        B = P.shape[0]
        temb = self.time_embed(t)
        pn = torch.linalg.norm(P, dim=-1, keepdim=True)
        tok = torch.cat([P, P / pn.clamp(min=1e-8), torch.log(pn + 1e-6),
                         temb[:, None, :].expand(B, self.n_particles, -1)], dim=-1)
        h = self.tok_in(tok)
        for blk in self.blocks:
            h = blk(h, temb)
        return self.out(self.out_norm(h))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class PSpaceDDPM:
    MODEL_TAG = "pspace_ddpm"

    def __init__(self, cfg: Config, seed: int = -1, gammas=None):
        """gammas: explicit step sizes instead of the schedule built from cfg (load() passes the stored array)."""
        self.cfg = cfg
        self.device = torch.device(cfg.device_str)
        self.dtype = torch.float32
        self.seed = seed
        if seed >= 0:
            torch.manual_seed(seed)
        self.gammas = self._build_gamma_schedule() if gammas is None else torch.as_tensor(gammas, device=self.device, dtype=self.dtype).clone()
        # DDPM tables in float64 (1 - abar_1 = 1e-7 is below float32 resolution of abar), stored as float32; index t = 0..T
        beta = 2 * self.gammas.double()
        log_abar = torch.cat([torch.zeros(1, device=self.device, dtype=torch.float64), torch.cumsum(torch.log1p(-beta), 0)])
        one_m_abar = -torch.expm1(log_abar)
        self.sqrt_abar = torch.exp(0.5 * log_abar).to(self.dtype)
        self.sqrt_1m_abar = torch.sqrt(one_m_abar).to(self.dtype)
        b = torch.cat([torch.zeros(1, device=self.device, dtype=torch.float64), beta])               # b[t] = beta_t
        self.eps_coef = (b[1:] / torch.sqrt(one_m_abar[1:])).to(self.dtype)                        # [t-1]: beta_t / sqrt(1-abar_t)
        self.mean_scale = (1 / torch.sqrt(1 - b[1:])).to(self.dtype)                               # [t-1]: 1 / sqrt(1-beta_t)
        self.post_std = torch.sqrt(b[1:] * one_m_abar[:-1] / one_m_abar[1:]).to(self.dtype)        # [t-1]: 0 at t = 1
        s2, abar = cfg.data_std ** 2, torch.exp(log_abar)                                           # output preconditioning, index t
        self.c_skip = (torch.sqrt(one_m_abar) / (abar * s2 + one_m_abar)).to(self.dtype)
        self.c_out = torch.sqrt(abar * s2 / (abar * s2 + one_m_abar)).to(self.dtype)
        self.net = EpsNetwork(cfg).to(self.device)
        self.ema_net = None

    def _build_gamma_schedule(self):
        """Same as model_singular.DiffusionModel._build_gamma_schedule."""
        cfg = self.cfg
        linear = torch.linspace(cfg.gamma_min, cfg.gamma_max, cfg.t_steps - cfg.t_geom, device=self.device)
        k = torch.arange(cfg.t_geom, device=self.device, dtype=torch.float32)
        geometric = cfg.gamma_geom * cfg.gamma_geom_growth ** k
        return torch.cat([geometric, linear])

    def eps(self, net, P, t):
        """eps_theta(P, t) = c_skip(t) P + c_out(t) net(P, t/T) for integer times t (B,)."""
        return self.c_skip[t][:, None, None] * P + self.c_out[t][:, None, None] * net(P, t.to(self.dtype) / len(self.gammas))

    # -- loss ---------------------------------------------------------------
    def compute_loss(self, P0):
        """L_simple on a batch of rescaled data P0 = P / x (B, N, 3) (already on the device)."""
        B, T = P0.shape[0], len(self.gammas)
        t = torch.randint(1, T + 1, (B,), device=self.device)
        eps = torch.randn_like(P0)
        Pt = self.sqrt_abar[t][:, None, None] * P0 + self.sqrt_1m_abar[t][:, None, None] * eps
        return ((self.eps(self.net, Pt, t) - eps) ** 2).sum(dim=(1, 2)).mean()

    # -- training -----------------------------------------------------------
    def train(self, p_train, seed=-1, callback=None, ckpt_path=None):
        """Same loop as model_singular.DiffusionModel.train: AdamW + cosine annealing, grad clipping, EMA after every
        step, callback(epoch, avg_loss), model.pt rewritten every 10 epochs.  p_train: rescaled data P / x (N_events, N, 3)."""
        cfg = self.cfg
        if seed >= 0:
            torch.manual_seed(seed)
        data = p_train.to(self.device, self.dtype)
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
    def sample_from(self, P, t, net=None):
        """Reverse (ancestral DDPM) process from P (B, N, 3) at integer time t down to t = 0 (net: an EpsNetwork, default self.net)."""
        net = self.net if net is None else net
        P = P.to(self.device, self.dtype).clone()
        for s in range(t, 0, -1):
            eps = self.eps(net, P, torch.full((P.shape[0],), s, device=self.device, dtype=torch.long))
            P = (P - self.eps_coef[s - 1] * eps) * self.mean_scale[s - 1]
            if s > 1:
                P = P + self.post_std[s - 1] * torch.randn_like(P)
        return P

    @torch.no_grad()
    def sample(self, n_samples, seed=-1):
        """Rescaled events P / x (multiply by cfg.x for physical momenta)."""
        if seed >= 0:
            torch.manual_seed(seed)
        self.net.eval()
        P = torch.randn((n_samples, self.cfg.n_particles, 3), device=self.device, dtype=self.dtype)
        return self.sample_from(P, len(self.gammas))

    # -- checkpointing ------------------------------------------------------
    def save(self, path):
        torch.save({"model": self.MODEL_TAG, "config": asdict(self.cfg), "seed": self.seed,
                    "gammas": self.gammas.detach().cpu().clone(),
                    "state_dict": self.net.state_dict(),
                    "ema_state_dict": (self.ema_net.state_dict() if self.ema_net is not None else None)}, path)

    @classmethod
    def load(cls, path, device="cuda", weights="ema"):
        """weights = 'ema' (default) or 'raw'.  Runs with the schedule stored in the checkpoint."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("model") != cls.MODEL_TAG:
            raise ValueError(f"{path} is a '{ck.get('model')}' checkpoint, not '{cls.MODEL_TAG}'")
        known = {f.name for f in fields(Config)}
        cfg = Config(**{k: v for k, v in ck["config"].items() if k in known})
        cfg.device_str = device
        m = cls(cfg, gammas=ck["gammas"])
        sd = ck["ema_state_dict"] if weights == "ema" and ck.get("ema_state_dict") is not None else ck["state_dict"]
        m.net.load_state_dict(sd)
        m.net.eval()
        return m


def load_pspace(path, n=0):
    """p-space events (N_events, N, 3) from a .pt file; n > 0 keeps the first n events."""
    p = torch.load(path, map_location="cpu", weights_only=True).float()
    return p[:n] if n > 0 and p.shape[0] > n else p
