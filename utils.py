"""Phase-space utilities needed for all models
"""

import math
import torch


def get_device(device_str="auto"):
    """Select compute device: CUDA > MPS > CPU cascade."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def make_generator(seed, device):
    """Create a torch.Generator. seed < 0 gives non-deterministic seeding."""
    gen = torch.Generator(device=device)
    if seed < 0:
        gen.seed()
    else:
        gen.manual_seed(seed)
    return gen


def sample_qspace(nevents, nparticles, seed=-1, device=None, dtype=torch.float32):
    """RAMBO q-space sampling: isotropic directions with E = -ln(r1*r2)."""
    if device is None:
        device = get_device()

    generator = make_generator(seed, device)

    r = torch.rand((4, nevents, nparticles), device=device, dtype=dtype,
                    generator=generator)

    c = 2.0 * r[0] - 1.0                                        # cos(theta)
    phi = 2.0 * math.pi * r[1]

    E = -torch.log(r[2].clamp(min=1e-30)) - torch.log(r[3].clamp(min=1e-30))   # (nevents, nparticles), avoids NaNs from underflow
    sin_theta = torch.sqrt(torch.clamp(1.0 - c ** 2, min=0))
    px = E * sin_theta * torch.cos(phi)
    py = E * sin_theta * torch.sin(phi)
    pz = E * c

    qs = torch.empty((nevents, nparticles, 3), device=device, dtype=dtype)
    qs[..., 0] = px
    qs[..., 1] = py
    qs[..., 2] = pz
    return qs


def get_b_x_from_qs(qs, energy=1.0):
    """Return CM boost vector b and conformal parameter x for given q vectors."""
    Qs = qs.sum(dim=1)                                           # (nevents, 3)
    Q0s = torch.sum(torch.linalg.norm(qs, dim=2), dim=1)        # (nevents,)

    Ms2 = Q0s ** 2 - torch.linalg.norm(Qs, dim=1) ** 2
    Ms = torch.sqrt(Ms2)

    bs = -Qs / Ms[:, None]
    xs = energy / Ms
    return bs, xs


def Hmu(inputvecs, bs):
    """Lorentz boost map (conformal rescaling omitted -- applied externally)."""
    gammas = torch.sqrt(1 + torch.linalg.norm(bs, axis=1) ** 2)
    As = 1 / (1 + gammas)
    bdotvecs = torch.einsum("ac,abc->ab", bs, inputvecs)
    outputvecs = (
        inputvecs
        + bs[:, None, :] * torch.linalg.norm(inputvecs, dim=2)[:, :, None]
        + As[:, None, None] * bdotvecs[:, :, None] * bs[:, None, :]
    )
    return outputvecs


def qs_to_ps(qs, energy=1.0):
    """q-space to p-space conformal map."""
    bs, xs = get_b_x_from_qs(qs, energy=energy)
    return Hmu(qs, bs) * xs[:, None, None]


def ps_to_qs(ps, bs, xs):
    """Inverse map: p-space to q-space."""
    return Hmu(ps, -bs) / xs[:, None, None]


def min_pairwise_dot(momenta):
    """min{E_i E_j - p_i . p_j} over all pairs, for arbitrary particle count.

    Args:
        momenta: (N, P, 3) tensor of 3-momenta (massless, E=|p|).

    Returns:
        (N,) tensor.
    """
    E = torch.linalg.norm(momenta, dim=-1)                       # (N, P)
    EiEj = E[:, :, None] * E[:, None, :]                        # (N, P, P)
    dots = torch.einsum("bid,bjd->bij", momenta, momenta)        # (N, P, P)
    pipj = EiEj - dots                                           # (N, P, P)

    P = momenta.shape[1]
    mask = torch.eye(P, device=momenta.device, dtype=torch.bool)
    pipj[:, mask] = float("inf")

    return pipj.amin(dim=-1).amin(dim=-1)                        # (N,)
