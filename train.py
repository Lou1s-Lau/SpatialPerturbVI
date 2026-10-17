"""Train SpatialPerturbVI on one dataset.

Example:
    python train.py --config configs/d1_perturbfish.json --data perturbfish.h5ad --seed 123 --out results/d1
"""

import argparse
import json
import math
import os
import random
import sys

import numpy as np
import scanpy as sc
import scvi
import torch

from spatialperturbvi import SpatialPerturbVI

TRAIN_FRAC = 0.9  # scvi default train_size


def prepare(adata, data_cfg, model_cfg):
    """Select perturbations, register the AnnData object and return the model arguments."""
    key, control = data_cfg["labels_key"], data_cfg["control_label"]
    if data_cfg.get("min_cells", 0) > 0:
        counts = adata.obs[key].value_counts()
        valid = set(counts[counts >= data_cfg["min_cells"]].index)
        valid |= {p for p in data_cfg.get("always_keep", []) if p in counts.index}
        valid.add(control)
        adata = adata[adata.obs[key].isin(valid)].copy()
    adata.obs[key] = adata.obs[key].astype(str).astype("category")

    adata.uns["spatial_mean"] = adata.obs[["X", "Y"]].values.mean(axis=0)
    adata.uns["spatial_std"] = adata.obs[["X", "Y"]].values.std(axis=0)

    model_kwargs = dict(model_cfg)
    cont_covs = ["X", "Y"]
    section_key = data_cfg.get("section_key")
    if section_key is not None:
        # The section index goes into the third continuous covariate; it selects the
        # prior network of each cell and the section mask of the spatial loss.
        sections = adata.obs[section_key].astype(str).astype("category")
        adata.obs["batch_id"] = sections.cat.codes.astype(float)
        cont_covs.append("batch_id")
        model_kwargs["per_batch_prior"] = int(sections.cat.categories.size)

    SpatialPerturbVI.setup_anndata(
        adata,
        layer=data_cfg.get("layer"),
        labels_key=key,
        continuous_covariate_keys=cont_covs,
        is_count_data=model_kwargs.get("gene_likelihood", "zinb") == "zinb",
    )
    background_indices = np.where(adata.obs[key] == control)[0].tolist()
    target_indices = np.where(adata.obs[key] != control)[0].tolist()
    return adata, background_indices, target_indices, model_kwargs


def kl_warmup_epochs(n_background, n_target, batch_size, kl_warmup_steps):
    """Number of KL warm-up epochs for a given number of gradient steps.

    The contrastive data loader drops incomplete batches and cycles the smaller side,
    so an epoch has as many steps as the larger side has full batches, and a side with
    fewer training cells than ``batch_size`` would give no training steps at all.
    """
    n_bg_train, n_tg_train = int(TRAIN_FRAC * n_background), int(TRAIN_FRAC * n_target)
    b_bg, b_tg = n_bg_train // batch_size, n_tg_train // batch_size
    if min(b_bg, b_tg) == 0:
        small, n_small = ("target", n_tg_train) if b_tg == 0 else ("background", n_bg_train)
        sys.exit(f"batch_size={batch_size} exceeds the {small} training set ({n_small} cells); "
                 f"use a batch size of at most {min(n_bg_train, n_tg_train)}.")
    steps_per_epoch = max(b_bg, b_tg)
    return max(1, math.ceil(kl_warmup_steps / steps_per_epoch))


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
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["train"]
    max_epochs = args.epochs if args.epochs is not None else train_cfg["max_epochs"]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    scvi.settings.seed = args.seed

    adata, background_indices, target_indices, model_kwargs = prepare(
        sc.read_h5ad(args.data), data_cfg, model_cfg)
    model = SpatialPerturbVI(adata, **model_kwargs)
    warmup = kl_warmup_epochs(len(background_indices), len(target_indices),
                              train_cfg["batch_size"], train_cfg["kl_warmup_steps"])
    model.train(
        background_indices=background_indices,
        target_indices=target_indices,
        max_epochs=max_epochs,
        batch_size=train_cfg["batch_size"],
        accelerator=args.accelerator,
        plan_kwargs={"lr": train_cfg["lr"], "n_epochs_kl_warmup": warmup},
    )

    os.makedirs(args.out, exist_ok=True)
    adata.obsm["spatial_latent"] = model.get_latent_representation(adata, representation_kind="background")
    adata.obsm["perturbation_latent"] = model.get_latent_representation(adata, representation_kind="salient")
    adata.obs["y_prob"] = np.asarray(model.get_escape_probability()).ravel()
    model.save(os.path.join(args.out, "model_dir"), overwrite=True, save_anndata=False)
    adata.write_h5ad(os.path.join(args.out, "result.h5ad"))
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(dict(cfg, seed=args.seed), f, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
