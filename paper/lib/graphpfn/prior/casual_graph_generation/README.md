# casual_graph_generation

The similarity-threshold geometric-prior graph generation process from
`synthetic_priors_geometric_similarity.ipynb` (cell `[23]`), packaged as a clean class.

A structural causal model (a random MLP-DAG) jointly produces each node's features `X`,
its label `y`, and a geometric embedding `Phi`. Nodes are connected when their embeddings
are similar under a chosen kernel, so feature- and label-homophily emerge from shared
ancestry — no homophily knob.

```
pipeline:  sample_scm_geo  ->  forward_scm_geo  ->  discretise_labels
                                       |
                                 frame_embedding  ->  build_similarity  ->  sample_similarity_graph
```

## Usage

```python
from casual_graph_generation import CausalGraphGenerator, GraphConfig

gen = CausalGraphGenerator(n_nodes=64, similarity='cosine', sim_threshold=0.5)
data = gen.generate(seed=7)          # reproducible draw
A, y, X = data['A'], data['y'], data['X']   # adjacency, labels, features

# or pass a config object
cfg = GraphConfig(n_nodes=128, similarity='mlp', sim_threshold=0.1, normalize='zscore')
data = CausalGraphGenerator(cfg).generate(seed=0, drop_isolated=True)
```

The functional entry point `sample_geo_similarity_dataset(...)` is also exported and is
API-compatible with the notebook.

## Sampling hyperparameters (matching the nodepfn causal prior)

`sample_config` draws the SCM/dataset hyperparameters from the **same distributions the
nodepfn pretraining prior uses** (`nodepfn/scripts/model_configs.py:get_diff_causal`,
`get_flexible_categorical_config`, and `nodepfn/pretrain.py`), so the generated graphs
cover the region of hyperparameter space the model is pretrained on. It re-draws until the
SCM constraint `n_features + n_geo <= n_layers*hidden - 1` holds.

| field | nodepfn source | distribution |
|---|---|---|
| `n_layers` | `num_layers` | `meta_gamma(max_alpha=2, max_scale=3, lb=2)` |
| `hidden` | `prior_mlp_hidden_dim` | `meta_gamma(max_alpha=3, max_scale=100, lb=4)` |
| `n_geo` | `num_causes` | `meta_gamma(max_alpha=3, max_scale=7, lb=2)` |
| `drop_rate` | `prior_mlp_dropout_prob` | `meta_beta(scale=0.6, min=0.1, max=5.0)` |
| `n_features` | `num_features_used` | `uniform_int(1, 100)` |
| `n_classes` | `num_classes` | `uniform_int(2, 20)` |

The similarity-kernel fields (`similarity`, `sim_threshold`, `normalize`, `sim_out_dim`,
`sim_act`) and `n_nodes` (= seq_len) have no nodepfn counterpart, so they keep their
cell `[23]` dashboard ranges. The exact distribution specs live in
`sampling.PRIOR_DISTRIBUTIONS`; the meta-distribution samplers reproduce
`nodepfn/priors/differentiable_prior.py`.

```python
import numpy as np
from casual_graph_generation import sample_config, CausalGraphGenerator

cfg = sample_config(seed=0)                       # one random valid config
gen = CausalGraphGenerator(cfg)
data = gen.generate(seed=0)

# convenience: sample the config and build the generator in one step
gen = CausalGraphGenerator.sample(seed=0)

# pin any field while sampling the rest (e.g. fix the kernel to cosine)
cfg = sample_config(seed=0, similarity='cosine')

# share one rng stream across many draws, as the notebook's bulk run does
rng = np.random.default_rng(0)
configs = [sample_config(rng=rng) for _ in range(100_000)]
```

The widget ranges live in `sampling.SLIDER_RANGES`, `SIMILARITY_CHOICES`,
`FRAME_CHOICES`, and `SIM_ACT_CHOICES`.

## Key options (`GraphConfig`)

| field | meaning |
|---|---|
| `n_nodes`, `n_features`, `n_classes` | dataset shape |
| `n_geo`, `n_layers`, `hidden` | SCM geometry / MLP-DAG size |
| `drop_rate` | SCM edge dropout (`None` → sampled) |
| `similarity` | `'cosine'` \| `'bilinear'` \| `'mlp'` \| callable |
| `sim_threshold` | tau — connect if similarity > tau |
| `normalize` | embedding frame: `activation` \| `none` \| `center` \| `minmax` \| `zscore` \| `rank` |
| `sim_out_dim`, `sim_hidden`, `sim_layers`, `sim_act`, `sim_scale` | bilinear / MLP kernel params |

`generate()` returns a dict with `X, y, y_hat, boundaries, A, S, Phi, Phi_framed, scm,
sim, sim_params, similarity, sim_threshold, normalize`.

Output is bit-for-bit identical to the notebook under the same seed.
