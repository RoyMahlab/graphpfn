"""Default hyperparameters, activation tables, and the TNLU sampler.

These mirror the toy defaults used in the synthetic-priors notebook (small enough
that the resulting DAGs and graphs stay easy to plot).
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch

# --- Toy hyperparameters (small so DAGs and graphs stay easy to plot) ---
N_FEATURES = 10      # number of observed feature columns
N_CLASSES = 3        # number of discrete classes for the supervised task
SCM_LAYERS = 4       # depth of the SCM's MLP-DAG
SCM_HIDDEN = 16      # width of each layer in the SCM
N_NODES_GRAPH = 64   # nodes in a graph dataset

# --- Number of geometric neurons used to embed nodes for the topology ---
N_GEO = 3            # |N_g| — chosen so the cosine degree calibration applies
SIM_THRESHOLD = 0.5  # tau — connect nodes whose similarity exceeds this


def tnlu(max_mean: float, min_mean: float, add_min: float = 0.0, do_round: bool = False) -> float:
    """Truncated-Normal Log-Uniform — TabPFN Table 5."""
    log_lo, log_hi = math.log(min_mean), math.log(max_mean)
    mu = math.exp(np.random.uniform(log_lo, log_hi))
    sigma = math.exp(np.random.uniform(log_lo, log_hi))
    v = -1.0
    while v < 0:
        v = float(np.random.normal(mu, sigma))
    if do_round:
        v = int(round(v))
    return v + add_min


ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    'tanh':       torch.tanh,
    'leaky_relu': torch.nn.functional.leaky_relu,
    'elu':        torch.nn.functional.elu,
    'identity':   lambda x: x,
}


# The similarity MLP uses only SYMMETRIC (odd) activations: a(-x) = -a(x). Oddness sends a
# centred embedding to a (roughly) centred feature cloud at every layer, so cosine in the
# feature space stays meaningful with NO offset correction. tanh/identity are odd;
# sin/erf/asinh extend the menu.
SYMMETRIC_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    'identity': lambda x: x,
    'tanh':     torch.tanh,
    'sin':      torch.sin,
    'erf':      torch.erf,
    'asinh':    torch.asinh,
}
