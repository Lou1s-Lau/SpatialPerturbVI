# SpatialPerturbVI

SpatialPerturbVI is a conditional variational autoencoder for spatially resolved CRISPR screens profiled at single-cell resolution. The expression of each cell combines the effect of its perturbation with variation that follows its position in the tissue; SpatialPerturbVI separates the two. It extends [ContrastiveVI+](https://github.com/insitro/contrastive_vi_plus), whose code it builds on.

- **Background latent space `z`.** Its prior is centered on a function of the spatial coordinates, with one network per tissue section.
- **Salient latent space `s`.** Its prior mixes a perturbation component, centered on a function of the perturbation label and the coordinates, and a basal component. The mixture weight is a per-cell gate `y` that marks perturbed cells that escaped their perturbation. Control cells are decoded with the basal state.
- **Additive decoder `f_z(z) + f_s(s)`.** The contribution of each latent space to every gene can be read out separately, which gives counterfactual perturbation signatures.

Coordinates and perturbation labels enter only the priors and the auxiliary losses; the encoders read expression alone.

## Installation

```bash
git clone https://github.com/Lou1s-Lau/SpatialPerturbVI.git
cd SpatialPerturbVI
pip install -r requirements.txt
```

Tested with Python 3.11, PyTorch 2.14 and scvi-tools 1.4.0. A GPU is recommended.

## Input

An AnnData object (`.h5ad`) with

| Field | Content |
|---|---|
| `adata.layers["counts"]` (or `adata.X`) | counts (ZINB likelihood) or normalized expression (Gaussian likelihood) |
| `obs["perturbation"]` | perturbation of each cell, including a control label |
| `obs["X"]`, `obs["Y"]` | spatial coordinates |
| `obs[section_key]` (optional) | tissue section of each cell, for screens with several sections |

## Usage

Train a model with one of the configurations in `configs/`:

```bash
python train.py --config configs/d1_perturbfish.json --data perturbfish.h5ad --seed 123 --out results/d1
```

This writes the trained model to `results/d1/model_dir` and an AnnData object `results/d1/result.h5ad` with the background latent (`obsm["spatial_latent"]`), the salient latent (`obsm["perturbation_latent"]`) and the gate (`obs["y_prob"]`).

Compute the counterfactual signature of every perturbation:

```bash
python compute_signatures.py --run results/d1
```

For every cell, the model decodes the expression expected without perturbation from the cell's own background latent, with the salient latent set to the basal state. The signature of a perturbation is the mean difference between observed and counterfactual expression on a logarithmic scale, relative to the same difference in control cells of the same section. The result is written to `results/d1/signatures.csv` (perturbations by genes).

The same steps from Python:

```python
import numpy as np
import scanpy as sc
from spatialperturbvi import SpatialPerturbVI, counterfactual_signatures

adata = sc.read_h5ad("screen.h5ad")
adata.uns["spatial_mean"] = adata.obs[["X", "Y"]].values.mean(axis=0)
adata.uns["spatial_std"] = adata.obs[["X", "Y"]].values.std(axis=0)
SpatialPerturbVI.setup_anndata(
    adata, layer="counts", labels_key="perturbation", continuous_covariate_keys=["X", "Y"]
)
model = SpatialPerturbVI(
    adata, n_hidden=128, n_background_latent=15, n_salient_latent=48, n_layers=2,
    mmd_penalty=1.0, adv_penalty=5.0, cls_penalty=6.0,
    spatial_contrastive_weight=8.0, spatial_repulsive_weight=0.5, spatial_knn_k=8,
)
background_indices = np.where(adata.obs["perturbation"] == "Control")[0].tolist()
target_indices = np.where(adata.obs["perturbation"] != "Control")[0].tolist()
model.train(
    background_indices=background_indices, target_indices=target_indices,
    max_epochs=300, batch_size=2048, plan_kwargs={"lr": 2e-3, "n_epochs_kl_warmup": 60},
)

z = model.get_latent_representation(adata, representation_kind="background")
s = model.get_latent_representation(adata, representation_kind="salient")
y = model.get_escape_probability(adata)
signatures = counterfactual_signatures(model, adata, control_label="Control")
```

For screens with several tissue sections, add the section index as a third continuous covariate and give each section its own prior network for `z`, as `train.py` does with `section_key` (see `configs/d3_liver.json`).

## Datasets

| Config | Dataset | Likelihood |
|---|---|---|
| `configs/d1_perturbfish.json` | Perturb-FISH screen of A375 melanoma xenografts (Binan et al., *Cell* 2025) | ZINB |
| `configs/d2_stereoseq.json` | Stereo-seq Spatial Perturb-seq screen of the mouse hippocampus, neurons of chip 2 (Shen et al., *Nat. Commun.* 2026; GEO GSE274447) | ZINB |
| `configs/d3_liver.json` | Perturb-Multimodal screen of mouse hepatocytes, seven sections (Saunders et al., *Cell* 2025; GEO GSE275483) | Gaussian |

The datasets are available from the original studies. The selection of cells, genes and perturbations is described in the Methods of the paper. For D3, expression and coordinates are centered within each section.

## Citation

Yusen Liu, Rui Wang and Hao Chen. SpatialPerturbVI: disentangling perturbation effects from tissue context in spatially resolved CRISPR screens.

## License

SpatialPerturbVI builds on [ContrastiveVI+](https://github.com/insitro/contrastive_vi_plus) (Copyright (C) 2024 Insitro, Inc.) and is distributed under the same license, the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). See `LICENSE`.
