"""Coordinate frames for the geometric embedding Phi.

Similarities are translation-sensitive, so where Phi sits matters. The 'activation'
frame removes the bias one-sided activations inject *analytically* (data-free) via
Gauss-Hermite moment propagation through the SCM; the other frames are empirical
normalisations kept for contrast.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .constants import ACTIVATIONS
from .scm import SCMGeo

# 64-point Gauss-Hermite rule for the weight e^{-x^2}: exact for smooth a, ~0.6% relative
# error on kinked activations (leaky_relu) — negligible for a mean-offset correction.
_gh = np.polynomial.hermite.hermgauss(64)
GH_NODES = torch.tensor(_gh[0], dtype=torch.float32)
GH_WEIGHTS = torch.tensor(_gh[1], dtype=torch.float32)


def activation_moments(
    act_name: str, mu: torch.Tensor, var: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """E[a(Z)] and Var[a(Z)] for Z ~ N(mu, var), elementwise, via Gauss-Hermite."""
    a = ACTIVATIONS[act_name]
    sigma = var.clamp_min(0.0).sqrt()
    # quadrature nodes  t_i = mu + sigma * sqrt(2) * xi_i   (broadcast over a trailing axis)
    pts = mu.unsqueeze(-1) + sigma.unsqueeze(-1) * (math.sqrt(2.0) * GH_NODES)
    fa = a(pts)
    norm = GH_WEIGHTS.sum()                       # = sqrt(pi);  (1/sqrt(pi)) * sum w_i = 1
    m1 = (fa * GH_WEIGHTS).sum(-1) / norm
    m2 = (fa * fa * GH_WEIGHTS).sum(-1) / norm
    return m1, (m2 - m1 * m1).clamp_min(0.0)


def propagate_moments(scm: SCMGeo) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-stored-layer (mean, var) of post-activations; input ~ N(0,1) per coord.

    Mean-field (independence-across-coordinates) approximation:
        pre_mean = W @ m_prev ;  pre_var = (W*W) @ v_prev + 0.1**2 .
    Returns means, vars each of shape (L, H).
    """
    mean = torch.zeros(scm.hidden)
    var = torch.ones(scm.hidden)
    means, varis = [], []
    for W in scm.weights:
        pre_mean = mean @ W.T                      # (H,)  == W @ mean
        pre_var = var @ (W * W).T + 0.1 ** 2       # eps ~ N(0, 0.1^2)
        mean, var = activation_moments(scm.activation_name, pre_mean, pre_var)
        means.append(mean)
        varis.append(var)
    return torch.stack(means), torch.stack(varis)  # (L, H), (L, H)


def analytic_offsets(scm: SCMGeo) -> torch.Tensor:
    """Predicted per-geometric-neuron activation offset c_d (the bias to subtract)."""
    means, _ = propagate_moments(scm)
    return means[scm.geo_idx[:, 0], scm.geo_idx[:, 1]]   # (n_geo,)


def frame_embedding(Phi: torch.Tensor, scm: SCMGeo, mode: str = 'activation') -> torch.Tensor:
    """Place Phi in the coordinate frame used for the similarity.

    'activation' (default): subtract the ANALYTIC activation offset c_d — a data-free
        origin correction from the known nonlinearity; no per-axis rescaling. No-op for
        tanh/identity. 'center': subtract the EMPIRICAL per-coordinate mean of Phi.
        'none': raw. 'minmax'/'zscore'/'rank': per-coordinate normalisations (for contrast).
    """
    if mode == 'activation':
        return Phi - analytic_offsets(scm).unsqueeze(0)
    if mode == 'center':
        return Phi - Phi.mean(dim=0, keepdim=True)
    if mode == 'none':
        return Phi
    if mode == 'minmax':
        lo = Phi.min(dim=0, keepdim=True).values
        hi = Phi.max(dim=0, keepdim=True).values
        return (Phi - lo) / (hi - lo).clamp_min(1e-12)
    if mode == 'zscore':
        mu = Phi.mean(dim=0, keepdim=True)
        std = Phi.std(dim=0, keepdim=True).clamp_min(1e-12)
        return (Phi - mu) / std
    if mode == 'rank':
        n = Phi.shape[0]
        order = Phi.argsort(dim=0)
        ranks = torch.empty_like(order, dtype=torch.float32)
        arange = torch.arange(n, dtype=torch.float32).unsqueeze(1).expand_as(order)
        ranks.scatter_(0, order, arange)
        return ranks / max(n - 1, 1)
    raise ValueError(f'unknown frame mode: {mode!r}')
