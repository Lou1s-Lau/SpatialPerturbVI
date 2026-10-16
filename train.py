"""Train SpatialPerturbVI on one dataset.

Example:
    python train.py --config configs/d1_perturbfish.json --data perturbfish.h5ad \
        --seed 123 --out results/d1
"""

import argparse
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import scanpy as sc  # noqa: E402
import scvi  # noqa: E402
import torch  # noqa: E402

from spatialperturbvi import SpatialPerturbVI  # noqa: E402


def load_data(path: str, cfg: dict):
    adata = sc.read_h5ad(path)
    key, control = cfg["perturbation_key"], cfg["control_label"]
    counts = adata.obs[key].astype(str).value_counts()
    keep = set(counts[counts >= cfg.get("min_cells", 0)].index)
    keep |= {p for p in cfg.get("always_keep", []) if p in counts.index}
    keep.add(control)
    adata = adata[adata.obs[key].astype(str).isin(keep)].copy()
    adata.obs[key] = adata.obs[key].astype(str).astype("category")
    return adata


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True, help="AnnData (.h5ad) with expression and coordinates")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=None, help="overrides the config")
    parser.add_argument("--accelerator", default="auto")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    data_cfg, model_cfg, train_cfg = cfg["data"], dict(cfg["model"]), dict(cfg["train"])
    if args.epochs is not None:
        train_cfg["max_epochs"] = args.epochs

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    scvi.settings.seed = args.seed

    adata = load_data(args.data, data_cfg)
    SpatialPerturbVI.setup_anndata(
        adata,
        perturbation_key=data_cfg["perturbation_key"],
        control_label=data_cfg["control_label"],
        x_key=data_cfg.get("x_key", "X"),
        y_key=data_cfg.get("y_key", "Y"),
        section_key=data_cfg.get("section_key"),
        layer=data_cfg.get("layer"),
        is_count_data=model_cfg.get("likelihood", "zinb") == "zinb",
    )
    model = SpatialPerturbVI(adata, **model_cfg)
    print(model)
    model.train(accelerator=args.accelerator, **train_cfg)

    os.makedirs(args.out, exist_ok=True)
    model.save(os.path.join(args.out, "model"), overwrite=True, save_anndata=False)
    adata.obsm["X_s"] = model.get_latent_representation(kind="salient")
    adata.obsm["X_z"] = model.get_latent_representation(kind="background")
    adata.obs["gate"] = model.get_gate()
    adata.write_h5ad(os.path.join(args.out, "adata.h5ad"))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
