"""Compute counterfactual perturbation signatures from a trained model.

Example:
    python compute_signatures.py --run results/d1
"""

import argparse
import json
import os

import scanpy as sc
import torch

from spatialperturbvi import SpatialPerturbVI, counterfactual_signatures
from train import prepare


def load_run(run: str):
    """Rebuild the model of a train.py output directory and load its weights."""
    with open(os.path.join(run, "config.json")) as f:
        cfg = json.load(f)
    adata, _, _, model_kwargs = prepare(sc.read_h5ad(os.path.join(run, "result.h5ad")), cfg["data"], cfg["model"])
    model = SpatialPerturbVI(adata, **model_kwargs)
    checkpoint = torch.load(os.path.join(run, "model_dir", "model.pt"), map_location="cpu", weights_only=False)
    state = {k: v for k, v in checkpoint["model_state_dict"].items() if k != "pyro_param_store"}
    model.module.load_state_dict(state)
    model.module.eval()
    model.is_trained_ = True
    return model, adata, cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="output directory of train.py")
    parser.add_argument("--out", default=None, help="CSV file (default: <run>/signatures.csv)")
    parser.add_argument("--min-controls", type=int, default=50)
    args = parser.parse_args()

    model, adata, cfg = load_run(args.run)
    signatures = counterfactual_signatures(
        model, adata, control_label=cfg["data"]["control_label"], section_key=cfg["data"].get("section_key"),
        min_controls=args.min_controls)
    out = args.out or os.path.join(args.run, "signatures.csv")
    signatures.to_csv(out)
    print(f"{signatures.shape[0]} signatures over {signatures.shape[1]} genes -> {out}")


if __name__ == "__main__":
    main()
