"""Counterfactual perturbation signatures."""

from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from anndata import AnnData


@torch.no_grad()
def counterfactual_expression(model, adata: Optional[AnnData] = None, batch_size: int = 4096) -> np.ndarray:
    """Expression expected for every cell without perturbation.

    Decoded from the cell's own background latent (posterior mean) with the salient
    latent set to the basal state, without sampling. For the ZINB likelihood this is
    ``(1 - pi) * softmax(logits)`` (to be multiplied by a library size); for the
    Gaussian likelihood it is the decoded mean.
    """
    module = model.module
    module.eval()
    z = model.get_latent_representation(adata, representation_kind="background")
    device = module.basal_mean.device
    out = []
    for i in range(0, len(z), batch_size):
        zt = torch.as_tensor(z[i:i + batch_size], dtype=torch.float32, device=device)
        st = module.basal_mean.reshape(1, -1).expand(zt.shape[0], -1)
        h = module.decoder_s_fc(st)
        logits = module.decoder_z_out(zt) + module.decoder_s_out(h)
        if module.gene_likelihood == "normal":
            out.append(logits.cpu().numpy())
        else:
            pi = torch.sigmoid(module.px_dropout_decoder(torch.cat([zt, h], -1)))
            out.append(((1 - pi) * torch.softmax(logits, -1)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def counterfactual_signatures(
    model,
    adata: Optional[AnnData] = None,
    control_label: str = "Control",
    section_key: Optional[str] = None,
    min_controls: int = 50,
) -> pd.DataFrame:
    """Signature of every perturbation (perturbations by genes).

    The residual of a cell is its observed expression minus its counterfactual
    expression on a logarithmic scale: for counts, both are scaled to the median
    library size and transformed with ``log(1 + x)``; normalized expression is used
    as is. The signature of a perturbation is the mean residual of its cells minus
    the mean residual of the control cells of the same section (all control cells if
    the section has fewer than ``min_controls``), averaged over sections with weights
    proportional to the number of perturbed cells.
    """
    adata = model._validate_anndata(adata)
    args = model.adata_manager.registry["setup_args"]
    x = adata.layers[args["layer"]] if args["layer"] is not None else adata.X
    x = np.asarray(x.toarray() if sp.issparse(x) else x, dtype=np.float64)
    expected = counterfactual_expression(model, adata)
    if model.module.gene_likelihood == "normal":
        residual = x - expected
    else:
        library = x.sum(1)
        median = np.median(library)
        residual = np.log1p(x / library[:, None] * median) - np.log1p(expected * median)

    labels = adata.obs[args["labels_key"]].astype(str).to_numpy()
    if section_key is not None:
        sections = adata.obs[section_key].astype(str).to_numpy()
    else:
        sections = np.zeros(adata.n_obs, dtype=str)
    control = labels == control_label

    signatures = {}
    for pert in sorted(set(labels) - {control_label}):
        total, weight = np.zeros(adata.n_vars), 0
        for section in np.unique(sections[labels == pert]):
            cells = (labels == pert) & (sections == section)
            ref = control & (sections == section)
            if ref.sum() < min_controls:
                ref = control
            total += cells.sum() * (residual[cells].mean(0) - residual[ref].mean(0))
            weight += cells.sum()
        signatures[pert] = total / weight
    return pd.DataFrame(signatures, index=adata.var_names).T
