"""Counterfactual perturbation signatures."""

from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from .model import SpatialPerturbVI


def counterfactual_signatures(
    model: SpatialPerturbVI,
    adata: Optional[AnnData] = None,
    min_controls: int = 50,
) -> pd.DataFrame:
    """Signature of every perturbation (perturbations by genes).

    For every cell, the model decodes the expression expected without perturbation
    from the cell's own background latent with the salient latent set to the basal
    state. The residual of a cell is its observed expression minus this
    counterfactual on a logarithmic scale: for counts, both are scaled to the median
    library size and transformed with ``log(1 + x)``; normalized expression is used
    as is. The signature of a perturbation is the mean residual of its cells minus
    the mean residual of the control cells of the same section (all control cells
    if the section has fewer than ``min_controls``), averaged over sections with
    weights proportional to the number of perturbed cells.
    """
    adata = model._validate_anndata(adata)
    args = model.adata_manager.registry["setup_args"]

    x = adata.layers[args["layer"]] if args["layer"] is not None else adata.X
    x = np.asarray(x.toarray() if sp.issparse(x) else x, dtype=np.float64)
    expected = model.get_counterfactual_expression(adata).astype(np.float64)
    if model.module.likelihood == "zinb":
        library = x.sum(1)
        median = np.median(library)
        residual = np.log1p(x / library[:, None] * median) - np.log1p(expected * median)
    else:
        residual = x - expected

    labels = adata.obs[args["perturbation_key"]].astype(str).to_numpy()
    if args["section_key"] is not None:
        sections = adata.obs[args["section_key"]].astype(str).to_numpy()
    else:
        sections = np.zeros(adata.n_obs, dtype=str)
    control = labels == args["control_label"]

    signatures = {}
    for pert in sorted(set(labels) - {args["control_label"]}):
        total, weight = np.zeros(adata.n_vars), 0
        for section in np.unique(sections[labels == pert]):
            cells = (labels == pert) & (sections == section)
            ref = control & (sections == section)
            if ref.sum() < min_controls:
                ref = control
            diff = residual[cells].mean(0) - residual[ref].mean(0)
            total += cells.sum() * diff
            weight += cells.sum()
        signatures[pert] = total / weight
    return pd.DataFrame(signatures, index=adata.var_names).T
