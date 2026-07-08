"""The causal graph generator.

`CausalGraphGenerator` packages the similarity-threshold graph generation process from
the notebook dashboard (cell [23]) into a configurable class with a single `generate()`
method. It samples a structural causal model, runs it forward to produce node features,
labels, and a geometric embedding, then connects nodes whose embeddings are similar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from .constants import (
    N_CLASSES,
    N_FEATURES,
    N_GEO,
    N_NODES_GRAPH,
    SCM_HIDDEN,
    SCM_LAYERS,
    SIM_THRESHOLD,
)
from .moments import frame_embedding
from .scm import discretise_labels, forward_scm_geo, sample_scm_geo
from .similarity import build_similarity


def sample_similarity_graph(
    Phi_framed: torch.Tensor,
    sim: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    threshold: float = SIM_THRESHOLD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adjacency from a similarity kernel: A[u,v] = 1 iff sim(u,v) > threshold. Returns (A, S)."""
    S = sim(Phi_framed, Phi_framed)                  # (n, n), symmetric, diag = 1
    A = (S > threshold).float()
    A.fill_diagonal_(0.0)
    return A, S


@dataclass
class GraphConfig:
    """Hyperparameters for one similarity-threshold graph (defaults match the notebook)."""

    n_nodes:       int = N_NODES_GRAPH
    n_features:    int = N_FEATURES
    n_classes:     int = N_CLASSES
    n_geo:         int = N_GEO
    n_layers:      int = SCM_LAYERS
    hidden:        int = SCM_HIDDEN
    drop_rate:     Optional[float] = None
    similarity:    str = 'cosine'          # 'cosine' | 'bilinear' | 'mlp' | callable
    sim_threshold: float = SIM_THRESHOLD   # tau
    normalize:     str = 'activation'      # embedding frame (see frame_embedding)
    sim_out_dim:   Optional[int] = None
    sim_hidden:    int = 16
    sim_layers:    int = 2
    sim_act:       str = 'tanh'
    sim_scale:     float = 1.0

    def is_valid(self) -> bool:
        """The SCM requires n_features + n_geo <= n_layers * hidden - 1."""
        return self.n_features + self.n_geo <= self.n_layers * self.hidden - 1


class CausalGraphGenerator:
    """Generate node-classification datasets whose topology is a similarity-threshold graph.

    Example
    -------
    >>> gen = CausalGraphGenerator(n_nodes=64, similarity='cosine', sim_threshold=0.5)
    >>> data = gen.generate(seed=7)
    >>> data['A'].shape, data['y'].shape
    """

    def __init__(self, config: GraphConfig | None = None, **overrides):
        if config is None:
            config = GraphConfig(**overrides)
        elif overrides:
            config = GraphConfig(**{**config.__dict__, **overrides})
        self.config = config

    @classmethod
    def sample(cls, rng=None, seed: int | None = None, **fixed) -> 'CausalGraphGenerator':
        """Build a generator whose hyperparameters are sampled over the cell [23] widget
        ranges (see :func:`casual_graph_generation.sampling.sample_config`). Pin any field
        via keyword (e.g. ``similarity='cosine'``)."""
        from .sampling import sample_config
        return cls(sample_config(rng=rng, seed=seed, **fixed))

    def generate(self, seed: int | None = None, drop_isolated: bool = False) -> dict:
        """Sample one dataset.

        Parameters
        ----------
        seed : optional int
            If given, seeds torch and numpy for a fully reproducible draw.
        drop_isolated : bool
            If True, remove degree-0 nodes from the returned graph (post-processing).

        Returns
        -------
        dict with keys:
            X, y, y_hat, boundaries, A, S, Phi, Phi_framed, scm, sim, sim_params,
            similarity, sim_threshold, normalize.
        """
        cfg = self.config
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed % (2 ** 32))

        scm = sample_scm_geo(
            n_features=cfg.n_features, n_geo=cfg.n_geo,
            n_layers=cfg.n_layers, hidden=cfg.hidden, drop_rate=cfg.drop_rate,
        )
        X, y_hat, Phi = forward_scm_geo(scm, cfg.n_nodes)
        y, boundaries = discretise_labels(y_hat, n_classes=cfg.n_classes)

        Phi_framed = frame_embedding(Phi, scm, cfg.normalize)
        sim, sim_params = build_similarity(
            cfg.similarity, in_dim=cfg.n_geo, sim_out_dim=cfg.sim_out_dim,
            sim_hidden=cfg.sim_hidden, sim_layers=cfg.sim_layers,
            sim_act=cfg.sim_act, sim_scale=cfg.sim_scale,
        )
        A, S = sample_similarity_graph(Phi_framed, sim, threshold=cfg.sim_threshold)

        if drop_isolated:
            keep = A.sum(dim=1) > 0
            A = A[keep][:, keep]
            S = S[keep][:, keep]
            X, y = X[keep], y[keep]
            y_hat = y_hat[keep]
            Phi, Phi_framed = Phi[keep], Phi_framed[keep]

        return {
            'X': X, 'y': y, 'y_hat': y_hat, 'boundaries': boundaries,
            'A': A, 'S': S, 'Phi': Phi, 'Phi_framed': Phi_framed,
            'scm': scm, 'sim': sim, 'sim_params': sim_params,
            'similarity': cfg.similarity if isinstance(cfg.similarity, str) else 'custom',
            'sim_threshold': cfg.sim_threshold, 'normalize': cfg.normalize,
        }


def sample_geo_similarity_dataset(
    n_nodes:       int = N_NODES_GRAPH,
    n_features:    int = N_FEATURES,
    n_classes:     int = N_CLASSES,
    n_geo:         int = N_GEO,
    n_layers:      int = SCM_LAYERS,
    hidden:        int = SCM_HIDDEN,
    drop_rate:     float | None = None,
    similarity:    str = 'cosine',
    sim_threshold: float = SIM_THRESHOLD,
    normalize:     str = 'activation',
    sim_out_dim:   int | None = None,
    sim_hidden:    int = 16,
    sim_layers:    int = 2,
    sim_act:       str = 'tanh',
    sim_scale:     float = 1.0,
) -> dict:
    """Functional wrapper around :class:`CausalGraphGenerator` (matches the notebook API)."""
    config = GraphConfig(
        n_nodes=n_nodes, n_features=n_features, n_classes=n_classes, n_geo=n_geo,
        n_layers=n_layers, hidden=hidden, drop_rate=drop_rate, similarity=similarity,
        sim_threshold=sim_threshold, normalize=normalize, sim_out_dim=sim_out_dim,
        sim_hidden=sim_hidden, sim_layers=sim_layers, sim_act=sim_act, sim_scale=sim_scale,
    )
    return CausalGraphGenerator(config).generate()
