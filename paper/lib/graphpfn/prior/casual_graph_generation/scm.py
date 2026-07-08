"""The structural causal model (SCM): a random MLP-DAG that jointly generates the
observed features X, the latent label score y_hat, and the geometric embedding Phi.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from .constants import (
    ACTIVATIONS,
    N_CLASSES,
    N_FEATURES,
    N_GEO,
    SCM_HIDDEN,
    SCM_LAYERS,
    tnlu,
)


@dataclass
class SCMGeo:
    """A sampled MLP-DAG together with the neuron positions read out as features,
    label, and geometric coordinates."""

    weights: list[torch.Tensor]   # layer-l weight matrices  (H x H)
    masks:   list[torch.Tensor]   # boolean edge masks       (H x H)
    activation_name: str
    feature_idx: torch.Tensor     # (k, 2)     — (layer, neuron) for each feature node
    label_idx:   int              # neuron index in the last layer
    geo_idx:     torch.Tensor     # (n_geo, 2) — (layer, neuron) for each geometric node
    drop_rate:   float
    hidden:      int = field(default=SCM_HIDDEN)
    n_layers:    int = field(default=SCM_LAYERS)

    @property
    def activation(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return ACTIVATIONS[self.activation_name]


def sample_scm_geo(
    n_features: int = N_FEATURES,
    n_geo:      int = N_GEO,
    n_layers:   int = SCM_LAYERS,
    hidden:     int = SCM_HIDDEN,
    drop_rate:  float | None = None,
) -> SCMGeo:
    """Draw a random MLP-DAG and assign feature / label / geometric neuron positions."""
    n_total = n_layers * hidden
    if n_features + n_geo > n_total - 1:
        raise ValueError(
            f'Need n_features + n_geo <= L*H - 1, got '
            f'{n_features} + {n_geo} > {n_total - 1} (L={n_layers}, H={hidden})'
        )

    # Edge-dropout rate ~ 0.9 * Beta(a, b), a,b ~ U(0.1, 5) unless an explicit value is passed.
    if drop_rate is None:
        a = float(np.random.uniform(0.1, 5.0))
        b = float(np.random.uniform(0.1, 5.0))
        drop_rate = float(np.random.beta(a, b)) * 0.9
    else:
        drop_rate = float(drop_rate)

    weights, masks = [], []
    for _ in range(n_layers):
        w_scale = tnlu(10.0, 0.01)
        W = torch.randn(hidden, hidden) * w_scale
        mask = (torch.rand(hidden, hidden) > drop_rate).float()
        weights.append(W * mask)
        masks.append(mask)

    # Label: a neuron in the last layer (math layer L; 0-indexed internally as n_layers-1).
    label_idx = int(torch.randint(0, hidden, (1,)).item())
    label_pair = (n_layers - 1, label_idx)

    # Enumerate ALL L*H post-activation neurons as (layer, neuron) pairs (0-indexed).
    # Layer index 0 here corresponds to math-layer 1 (the first post-activation layer).
    all_pairs = torch.stack(torch.meshgrid(
        torch.arange(n_layers), torch.arange(hidden), indexing='ij',
    ), dim=-1).reshape(-1, 2)                       # (L*H, 2)

    # Drop the label pair, then draw n_features + n_geo without replacement.
    label_row = (all_pairs[:, 0] == label_pair[0]) & (all_pairs[:, 1] == label_pair[1])
    candidates = all_pairs[~label_row]               # (L*H - 1, 2)
    perm = torch.randperm(candidates.shape[0])
    feature_idx = candidates[perm[:n_features]]                       # (k, 2)
    geo_idx     = candidates[perm[n_features:n_features + n_geo]]      # (n_geo, 2)

    act_name = str(np.random.choice(list(ACTIVATIONS)))

    return SCMGeo(
        weights=weights, masks=masks, activation_name=act_name,
        feature_idx=feature_idx, label_idx=label_idx, geo_idx=geo_idx,
        drop_rate=drop_rate, hidden=hidden, n_layers=n_layers,
    )


def forward_scm_geo(
    scm: SCMGeo, n_samples: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run n_samples i.i.d. inputs through the SCM and read out (X, y_hat, Phi)."""
    z = torch.randn(n_samples, scm.hidden)        # input layer ~ N(0, 1)
    activations_per_layer = [z]
    for W in scm.weights:
        eps = torch.randn(n_samples, scm.hidden) * 0.1   # per-layer noise ~ N(0, 0.1^2)
        z = scm.activation(z @ W.T + eps)
        activations_per_layer.append(z)

    # Stack the L post-activation layers: shape (n, L, H). Index 0 corresponds to math-layer 1.
    acts = torch.stack(activations_per_layer[1:], dim=1)

    # Gather feature / geometric values from arbitrary (layer, neuron) positions.
    X   = acts[:, scm.feature_idx[:, 0], scm.feature_idx[:, 1]]   # (n, k)
    Phi = acts[:, scm.geo_idx[:, 0],     scm.geo_idx[:, 1]]        # (n, n_geo)
    # Label sits in the last layer (acts[:, -1] == activations_per_layer[-1]).
    y_hat = acts[:, -1, scm.label_idx]                              # (n,)
    return X, y_hat, Phi


def discretise_labels(
    y_hat: torch.Tensor,
    n_classes: int = N_CLASSES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn the continuous label score into n_classes balanced-ish classes by quantile cuts."""
    n = y_hat.numel()
    sorted_y, _ = torch.sort(y_hat)
    boundary_indices = np.sort(
        np.random.choice(n, size=n_classes - 1, replace=False)
    )
    boundaries = sorted_y[boundary_indices]

    y = torch.zeros(n, dtype=torch.long)
    for B in boundaries:
        y = y + (y_hat > B).long()

    perm = torch.randperm(n_classes)
    y = perm[y]
    return y, boundaries
