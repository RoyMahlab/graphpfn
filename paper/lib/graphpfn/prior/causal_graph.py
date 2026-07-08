"""Causal-graph prior: adapts ``casual_graph_generation.CausalGraphGenerator`` to the
GraphPFN pretraining ``PriorDataset`` / ``PriorDatasetBatch`` interface.

This is a drop-in alternative to :class:`GraphPriorSampler`. It is selected in
``bin/graphpfn/pretrain.py`` via ``prior_method = "causal_graph"`` and exposes the same
iterator / DDP / ``sample_batch`` surface, so the rest of the training loop is unchanged.

Where the settings come from
----------------------------
* The *graph-generation* hyperparameters (SCM depth/width, number of causes, dropout,
  feature count, class count, similarity kernel) are drawn by
  ``casual_graph_generation.sample_config``, which already mirrors the causal prior's
  distributions.
* The *shared training* settings that also exist for the default prior -- the node-count
  range, ``train_ratio``, ``batch_size``, ``n_workers``, ``min_features`` and the class
  count -- are taken from the config so they match the existing prior as closely as
  possible. ``n_nodes`` (log-uniform-int) and ``train_ratio`` (uniform) are drawn with the
  same distributions the default ``pretrain.toml`` uses.

The RNG follows the same convention as :mod:`lib.graphpfn.prior.sampler`: all randomness
flows from the global torch / legacy-numpy RNG state, which ``delu.random.seed`` sets per
worker, so sampling is reproducible under the DataLoader worker seeding.
"""

from __future__ import annotations

from functools import partial
from typing import NotRequired, Self, TypedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from lib.graphpfn.prior.casual_graph_generation import CausalGraphGenerator, sample_config

from lib.util import TaskType, get_world_size, is_master_process

from .checks import SanityCheckError, check_dataset
from .postprocessing import drop_constant_features
from .prior_typings import PriorDataset, PriorDatasetBatch
from .sampler import (
    GraphPriorSampler,
    GraphPriorSamplerDDP,
    _identity,
    _pad_and_batch,
    _worker_init,
)


class RangeConfig(TypedDict):
    min: float
    max: float


class CausalPriorConfig(TypedDict):
    """Config for the causal-graph prior (the ``[causal_prior]`` config section)."""

    min_features: int  # discard datasets with fewer feature columns (matches sanity_check)
    max_num_classes: int  # class count is drawn uniform_int(2, max_num_classes) per batch
    n_nodes: RangeConfig  # log-uniform-int graph size, drawn once per batch
    train_ratio: RangeConfig  # uniform train fraction, drawn once per batch
    # Optionally pin any GraphConfig field (e.g. {"similarity": "cosine"}); otherwise the
    # similarity kernel and SCM hyperparameters are sampled by sample_config.
    fixed: NotRequired[dict]


# >>> Distribution helpers (mirror lib.graphpfn.prior.config sampling of the same fields)


def _sample_log_uniform_int(lo: int, hi: int) -> int:
    return int(round(float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))))


# >>> Dataset conversion


def _to_prior_dataset(
    data: dict,
    n_train_nodes: int,
    n_classes: int,
) -> PriorDataset:
    """Convert a CausalGraphGenerator draw into a PriorDataset.

    The task type follows the *requested* ``n_classes`` (binclass iff it is 2). Labels are
    relabelled to a contiguous ``0..k-1`` range; if a discretisation bin collapsed, the
    resulting class count is < ``n_classes`` and ``check_dataset`` rejects the draw so the
    caller redraws -- this matches the default prior, which also enforces an exact count.

    Features that are constant on the *training* split are dropped (as the default prior
    does): the LimiX preprocessor filters such columns internally, so leaving them in makes
    ``num_used_features`` disagree with ``features.shape[-1]`` and corrupts the model.
    """
    A = data["A"]  # (n, n) dense adjacency, diagonal already zeroed
    X = data["X"].to(torch.float32)  # (n, n_features)
    y = data["y"]  # (n,) long class ids

    # Drop columns that are constant across the training rows (matches graph_then_attributes).
    _, feature_mask = drop_constant_features(X[:n_train_nodes, :])
    X = X[:, feature_mask]

    _, y_contiguous = torch.unique(y, return_inverse=True)
    task_type = TaskType.BINCLASS if n_classes == 2 else TaskType.MULTICLASS

    src, dst = torch.nonzero(A, as_tuple=True)
    edges = torch.stack([src, dst], dim=0).to(torch.int64)

    return PriorDataset(
        features=X,
        labels=y_contiguous.to(torch.float32),
        edges=edges,
        n_train_nodes=n_train_nodes,
        task_type=task_type,
    )


def _sample_dataset_with_retry(
    config: CausalPriorConfig,
    n_nodes: int,
    n_train_nodes: int,
    n_classes: int,
    max_retries: int = 3,
) -> PriorDataset:
    min_features = config["min_features"]
    fixed = dict(config.get("fixed", {}))

    for attempt in range(max_retries):
        try:
            # Seed a per-draw Generator from the (delu-seeded) legacy RNG so the config
            # draw is reproducible per worker while staying decoupled per dataset.
            rng = np.random.default_rng(int(np.random.randint(0, 2**31 - 1)))
            cfg = sample_config(
                rng=rng, n_nodes=n_nodes, n_classes=n_classes, **fixed
            )
            data = CausalGraphGenerator(cfg).generate()
            dataset = _to_prior_dataset(data, n_train_nodes, n_classes)

            # Enforce the *requested* class count (and, via check_class_coverage, that both
            # the train and test split contain every class) so degenerate single-class
            # draws are redrawn instead of breaking classification metrics at eval time.
            check_dataset(
                features=dataset["features"],
                labels=dataset["labels"],
                n_train_nodes=n_train_nodes,
                task_type=dataset["task_type"],
                min_features=min_features,
                n_classes=n_classes
                if dataset["task_type"] == TaskType.MULTICLASS
                else None,
            )
            return dataset

        except SanityCheckError:
            if attempt == max_retries - 1:
                raise

    raise SanityCheckError(f"Failed after {max_retries} attempts")


# >>> Public functional API (mirrors lib.graphpfn.prior.sampler.sample_batch)


def sample_causal_batch(
    config: CausalPriorConfig,
    batch_size: int,
    device: torch.device,
) -> PriorDatasetBatch:
    """Sample a batch of causal-graph datasets and move it to ``device``.

    ``n_nodes``, ``train_ratio`` and the class count are drawn once per batch (shared
    across the batch, like the default prior's ``_shared_`` fields) so every dataset in
    the batch has a consistent split and task type.
    """
    while True:
        try:
            n_nodes = _sample_log_uniform_int(
                int(config["n_nodes"]["min"]), int(config["n_nodes"]["max"])
            )
            train_ratio = float(
                np.random.uniform(
                    config["train_ratio"]["min"], config["train_ratio"]["max"]
                )
            )
            n_train_nodes = int(n_nodes * train_ratio)
            n_classes = int(np.random.randint(2, int(config["max_num_classes"]) + 1))

            datasets = [
                _sample_dataset_with_retry(config, n_nodes, n_train_nodes, n_classes)
                for _ in range(batch_size)
            ]
            batch = _pad_and_batch(datasets)
        except SanityCheckError:
            continue

        batch["features"] = batch["features"].to(device)
        batch["labels"] = batch["labels"].to(device)
        batch["n_nodes"] = batch["n_nodes"].to(device)
        batch["n_features"] = batch["n_features"].to(device)

        return batch


# >>> Iterator interface (mirrors lib.graphpfn.prior.sampler.GraphPriorSampler)


class _CausalBatchIterableDataset(IterableDataset[PriorDatasetBatch]):
    def __init__(
        self,
        config: CausalPriorConfig,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.config = config
        self.batch_size = batch_size
        self.device = device

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> PriorDatasetBatch:
        return sample_causal_batch(self.config, self.batch_size, self.device)


class CausalGraphPriorSampler(GraphPriorSampler):
    """Iterator yielding batches of synthetic causal-graph datasets.

    Same interface as :class:`GraphPriorSampler`; only the per-batch generation differs.
    """

    def __init__(
        self,
        config: CausalPriorConfig,
        batch_size: int,
        seed: int | None = None,
        n_workers: int = 0,
        prefetch_factor: int | None = None,
        verbose: bool = True,
    ) -> None:
        if n_workers > 0 and seed is None:
            raise ValueError("seed is required when n_workers > 0")

        self.verbose = verbose

        dataset = _CausalBatchIterableDataset(config, batch_size, torch.device("cpu"))

        if n_workers == 0:
            self._iterator = iter(dataset)
        else:
            assert seed is not None
            loader = DataLoader(
                dataset=dataset,
                batch_size=1,
                num_workers=n_workers,
                worker_init_fn=partial(_worker_init, seed=seed),
                multiprocessing_context=torch.multiprocessing.get_context("spawn"),
                collate_fn=_identity,
                prefetch_factor=prefetch_factor,
            )
            self._iterator = iter(loader)


class CausalGraphPriorSamplerDDP(GraphPriorSamplerDDP):
    """DDP wrapper for :class:`CausalGraphPriorSampler` (mirrors GraphPriorSamplerDDP)."""

    def __init__(
        self,
        config: CausalPriorConfig,
        batch_size: int,
        seed: int | None = None,
        n_workers: int = 0,
        prefetch_factor: int | None = None,
        verbose: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.verbose = verbose

        if is_master_process():
            world_size = get_world_size()
            self.prior = CausalGraphPriorSampler(
                config=config,
                batch_size=world_size * batch_size,
                seed=seed,
                n_workers=world_size * n_workers,
                prefetch_factor=prefetch_factor,
                verbose=False,
            )
        else:
            self.prior = None
