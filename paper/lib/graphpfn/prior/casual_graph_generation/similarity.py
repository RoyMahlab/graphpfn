"""Similarity kernels that turn the geometric embedding into graph topology.

The cosine -> bilinear -> MLP ladder is an increasingly expressive source of topology
randomness, layered on top of and independent of the SCM.
"""
from __future__ import annotations

import math
from typing import Callable

import torch

from .constants import SYMMETRIC_ACTIVATIONS


def cosine_sim(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity, (n,m). Symmetric and = 1 on the diagonal for A==B."""
    An = A / A.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Bn = B / B.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return An @ Bn.T


def make_bilinear_sim(in_dim: int, out_dim: int | None = None, scale: float = 1.0):
    """Cosine under a random linear warp: sim(a,b) = cos(La, Lb), L ~ N(0, scale^2)."""
    out_dim = in_dim if out_dim is None else out_dim
    L = torch.randn(out_dim, in_dim) * scale

    def sim(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return cosine_sim(A @ L.T, B @ L.T)

    return sim, {'L': L, 'out_dim': out_dim}


def make_mlp_sim(in_dim: int, hidden: int = 16, out_dim: int | None = None,
                 n_layers: int = 2, act: str = 'tanh'):
    """Cosine in a random nonlinear feature space: sim(a,b) = cos(phi(a), phi(b)).

    phi is a random MLP independent of the SCM, with a SYMMETRIC (odd) hidden activation
    and a linear read-out. Oddness sends a centred embedding to a (roughly) centred feature
    cloud, so cosine in feature space needs no offset correction. The linear read-out makes
    n_layers=1 reduce exactly to the bilinear kernel, so the cosine -> bilinear -> MLP ladder
    stays continuous; n_layers>=2 is genuinely nonlinear. Weights are fan-in-scaled Gaussians.
    """
    if act not in SYMMETRIC_ACTIVATIONS:
        raise ValueError(
            f'similarity-MLP activation must be symmetric (odd); got {act!r}. '
            f'choose from {sorted(SYMMETRIC_ACTIVATIONS)}'
        )
    out_dim = in_dim if out_dim is None else out_dim
    dims = [in_dim] + [hidden] * max(n_layers - 1, 0) + [out_dim]
    Ws = [torch.randn(dims[i + 1], dims[i]) / math.sqrt(dims[i]) for i in range(len(dims) - 1)]
    a = SYMMETRIC_ACTIVATIONS[act]

    def phi(X: torch.Tensor) -> torch.Tensor:
        z = X
        for i, W in enumerate(Ws):
            z = z @ W.T
            if i < len(Ws) - 1:        # odd nonlinearity on hidden layers only; linear read-out
                z = a(z)
        return z

    def sim(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return cosine_sim(phi(A), phi(B))

    return sim, {'weights': Ws, 'act': act, 'out_dim': out_dim, 'phi': phi}


def build_similarity(spec, in_dim: int, **kw):
    """Return (sim_callable, params). spec in {'cosine','bilinear','mlp'} or a raw callable."""
    if callable(spec):
        return spec, {}
    if spec == 'cosine':
        return cosine_sim, {}
    if spec == 'bilinear':
        return make_bilinear_sim(in_dim, out_dim=kw.get('sim_out_dim'),
                                 scale=kw.get('sim_scale', 1.0))
    if spec == 'mlp':
        return make_mlp_sim(in_dim, hidden=kw.get('sim_hidden', 16),
                            out_dim=kw.get('sim_out_dim'),
                            n_layers=kw.get('sim_layers', 2), act=kw.get('sim_act', 'tanh'))
    raise ValueError(f'unknown similarity spec: {spec!r}')
