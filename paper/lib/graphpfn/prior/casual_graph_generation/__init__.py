"""Causal graph generation.

Similarity-threshold geometric-prior graphs: a structural causal model jointly produces
node features, labels, and a geometric embedding; nodes are connected when their
embeddings are similar under a chosen kernel (cosine / bilinear / MLP).

Extracted from the synthetic-priors notebook (graph generation in cell [23]).
"""
from __future__ import annotations

from .constants import (
    ACTIVATIONS,
    N_CLASSES,
    N_FEATURES,
    N_GEO,
    N_NODES_GRAPH,
    SCM_HIDDEN,
    SCM_LAYERS,
    SIM_THRESHOLD,
    SYMMETRIC_ACTIVATIONS,
    tnlu,
)
from .generator import (
    CausalGraphGenerator,
    GraphConfig,
    sample_geo_similarity_dataset,
    sample_similarity_graph,
)
from .moments import (
    activation_moments,
    analytic_offsets,
    frame_embedding,
    propagate_moments,
)
from .sampling import (
    FRAME_CHOICES,
    PRIOR_DISTRIBUTIONS,
    SIMILARITY_CHOICES,
    SIM_ACT_CHOICES,
    SLIDER_RANGES,
    sample_config,
)
from .scm import SCMGeo, discretise_labels, forward_scm_geo, sample_scm_geo
from .similarity import (
    build_similarity,
    cosine_sim,
    make_bilinear_sim,
    make_mlp_sim,
)

__all__ = [
    'CausalGraphGenerator',
    'GraphConfig',
    'sample_config',
    'PRIOR_DISTRIBUTIONS',
    'SLIDER_RANGES',
    'SIMILARITY_CHOICES',
    'FRAME_CHOICES',
    'SIM_ACT_CHOICES',
    'sample_geo_similarity_dataset',
    'sample_similarity_graph',
    'SCMGeo',
    'sample_scm_geo',
    'forward_scm_geo',
    'discretise_labels',
    'frame_embedding',
    'activation_moments',
    'propagate_moments',
    'analytic_offsets',
    'cosine_sim',
    'make_bilinear_sim',
    'make_mlp_sim',
    'build_similarity',
    'tnlu',
    'ACTIVATIONS',
    'SYMMETRIC_ACTIVATIONS',
    'N_FEATURES',
    'N_CLASSES',
    'N_GEO',
    'N_NODES_GRAPH',
    'SCM_HIDDEN',
    'SCM_LAYERS',
    'SIM_THRESHOLD',
]
