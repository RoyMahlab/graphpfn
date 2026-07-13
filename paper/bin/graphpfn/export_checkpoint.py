"""Convert a pretraining checkpoint into an eval-ready checkpoint.

Pretraining (``bin/graphpfn/pretrain.py``) writes ``checkpoint.pt`` with keys
``model`` / ``model_ema`` (the EMA weights are wrapped in an
``torch.optim.swa_utils.AveragedModel``, so their keys carry a ``module.``
prefix plus an ``n_averaged`` buffer). The evaluation loader
(``bin/graphpfn/evaluate.py``) instead expects a ``state_dict`` key. This
script bridges the two: it extracts the requested weights, strips the EMA
prefix, and saves ``{"state_dict": ...}``.

Usage:
    uv run bin/graphpfn/export_checkpoint.py \
        exp/graphpfn/pretrain/causal/pretrain/checkpoint.pt \
        checkpoints/graphpfn/causal-pretrain-ema.pt

    # use the raw (non-EMA) weights instead
    uv run bin/graphpfn/export_checkpoint.py IN OUT --weights model
"""

import argparse
from pathlib import Path

import torch


def convert(src: Path, dst: Path, weights: str = "model_ema") -> None:
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    if weights not in checkpoint:
        raise KeyError(
            f"'{weights}' not found in {src}; available keys: {list(checkpoint)}"
        )

    raw = checkpoint[weights]
    if weights == "model_ema":
        # AveragedModel wraps the module: keys are "module.<name>", plus an
        # "n_averaged" buffer that is not part of the model. Mirror the
        # stripping done in bin/graphpfn/pretrain.py when resuming.
        state_dict = {k[len("module.") :]: v for k, v in raw.items() if k.startswith("module.")}
    else:
        state_dict = dict(raw)

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict}, dst)
    print(f"step={checkpoint.get('step')} | {weights} tensors={len(state_dict)}")
    print(f"wrote {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="Path to the pretraining checkpoint.pt")
    parser.add_argument("dst", type=Path, help="Path to write the eval-ready checkpoint")
    parser.add_argument(
        "--weights",
        choices=["model_ema", "model"],
        default="model_ema",
        help="Which weights to export (default: model_ema, matching the released model)",
    )
    args = parser.parse_args()
    convert(args.src, args.dst, args.weights)


if __name__ == "__main__":
    main()
