# type: ignore
"""SpatialPerturbVI: ContrastiveVI+ with spatial priors and an additive decoder."""

import math
import os
import tempfile
import warnings
from functools import partial
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from anndata import AnnData
from .loss_recorder import LossOutput
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.dataloaders import AnnDataLoader
from scvi.distributions import ZeroInflatedNegativeBinomial
from scvi.model._utils import (
    _get_batch_code_from_category,
    _init_library_size,
    scrna_raw_counts_properties,
)
from scvi.model.base._de_core import _de_core
from scvi.module.base import BaseModuleClass, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, FCLayers
from torch.nn.functional import one_hot
from scvi.utils import setup_anndata_dsp
from torch import nn
from torch.distributions import Bernoulli, Normal
from torch.distributions import kl_divergence as kl

from .base_contrastive_model import BaseContrastiveModelClass
from .utils import gumbel_sigmoid, mmd

Number = Union[int, float]
torch.backends.cudnn.benchmark = True


class Classifier(nn.Module):
    """Basic fully-connected NN classifier.

    Parameters
    ----------
    n_input
        Number of input dimensions
    n_hidden
        Number of nodes in hidden layer(s). If `0`, the classifier only consists of a
        single linear layer.
    n_labels
        Numput of outputs dimensions
    n_layers
        Number of hidden layers. If `0`, the classifier only consists of a single
        linear layer.
    dropout_rate
        dropout_rate for nodes
    logits
        Return logits or not
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    activation_fn
        Valid activation function from torch.nn
    **kwargs
        Keyword arguments passed into :class:`~scvi.nn.FCLayers`.
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int = 256,
        n_labels: int = 5,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        logits: bool = False,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        activation_fn: nn.Module = nn.ReLU,
        **kwargs,
    ):
        super().__init__()
        self.logits = logits
        layers = []

        if n_hidden > 0 and n_layers > 0:
            layers.append(
                FCLayers(
                    n_in=n_input,
                    n_out=n_hidden,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    use_batch_norm=use_batch_norm,
                    use_layer_norm=use_layer_norm,
                    activation_fn=activation_fn,
                    **kwargs,
                )
            )
        else:
            n_hidden = n_input

        layers.append(nn.Linear(n_hidden, n_labels))

        if not logits:
            layers.append(nn.Softmax(dim=-1))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        """Forward computation."""
        return self.classifier(x)


class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal layer for adversarial training."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversal(nn.Module):
    """Gradient reversal module."""

    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class ContrastiveVIPlusModule(BaseModuleClass):
    """
    PyTorch module for Contrastive VI (Variational Inference).

    Args:
    ----
        n_input: Number of input genes.
        n_batch: Number of batches. If 0, no batch effect correction is performed.
        n_hidden: Number of nodes per hidden layer.
        n_background_latent: Dimensionality of the background latent space.
        n_salient_latent: Dimensionality of the salient latent space.
        n_layers: Number of hidden layers used for encoder and decoder NNs.
        dropout_rate: Dropout rate for neural networks.
        use_observed_lib_size: Use observed library size for RNA as scaling factor in
            mean of conditional distribution.
        library_log_means: 1 x n_batch array of means of the log library sizes.
            Parameterize prior on library size if not using observed library size.
        library_log_vars: 1 x n_batch array of variances of the log library sizes.
            Parameterize prior on library size if not using observed library size.
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 256,
        n_background_latent: int = 10,
        n_salient_latent: int = 10,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        learn_basal_mean: bool = True,
        mmd_penalty: float = 0,
        inference_strategy: Literal["marginalize", "gumbel_sigmoid"] = "marginalize",
        adv_penalty: float = 5.0,
        cls_penalty: float = 1.0,
        kl_y_weight: float = 1.0,
        # Spatial coordinate normalization stats
        spatial_mean: Optional[np.ndarray] = None,
        spatial_std: Optional[np.ndarray] = None,
        # Spatial contrastive loss
        spatial_contrastive_weight: float = 0.0,
        spatial_repulsive_weight: float = 0.0,
        spatial_knn_k: int = 5,
        # Likelihood: "zinb" (raw counts, default) or "normal" (Gaussian, for
        # scaled/log-normalized data that has no raw counts — e.g. Perturb-Multimodal
        # hepatocyte MERFISH). scGen-style Gaussian VAE on log-norm data.
        gene_likelihood: Literal["zinb", "normal"] = "zinb",
        learnable_px_var: bool = True,
        # Free bits (nats per dimension) for the KL of z; free_bits_z overrides free_bits.
        free_bits: float = 0.0,
        free_bits_z: Optional[float] = None,
        # Upper bound of the per-gene variance of the Gaussian likelihood.
        px_var_max: Optional[float] = None,
        # Weights of the KL terms of z and s.
        kl_z_weight: float = 1.0,
        kl_s_weight: float = 1.0,
        # Number of tissue sections with their own prior network (0: one shared network);
        # per_batch_prior_which selects the priors that are split ("z", "s" or "both").
        per_batch_prior: int = 0,
        per_batch_prior_which: str = "both",
        # Restrict the spatial contrastive pairs to cells of the same section.
        umap_same_batch: bool = False,
        # Use the spatial prior network for z (False: N(0, I) prior).
        use_z_prior_net: bool = True,
        # a and b of the kernel q = 1 / (1 + a * d^(2b)) in the spatial contrastive loss.
        umap_a: float = 1.0,
        umap_b: float = 1.0,
        # Include the coordinates in the input of the perturbation prior network of s.
        prior_s_use_xy: bool = True,
    ) -> None:
        super().__init__()
        self.free_bits_z = free_bits if free_bits_z is None else free_bits_z
        self.px_var_max = px_var_max
        self.n_input = n_input
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_hidden = n_hidden
        self.n_background_latent = n_background_latent
        self.n_salient_latent = n_salient_latent
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.latent_distribution = "normal"
        self.dispersion = "gene"
        self.px_r = torch.nn.Parameter(torch.randn(n_input))
        # Gaussian-likelihood params (used only when gene_likelihood == "normal").
        # px_log_var = per-gene log variance; init 0 → σ²=1 (scGen-style MSE).
        self.gene_likelihood = gene_likelihood
        self.px_log_var = torch.nn.Parameter(
            torch.zeros(n_input), requires_grad=learnable_px_var
        )
        self.use_observed_lib_size = use_observed_lib_size

        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )
            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )

        cat_list = [n_batch]
        # Background encoder.
        self.z_encoder = Encoder(
            n_input,
            n_background_latent,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=self.latent_distribution,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
            var_activation=None,
        )
        # Salient encoder.
        self.s_encoder = Encoder(
            n_input,
            n_salient_latent,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=self.latent_distribution,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
            var_activation=None,
        )
        # Library size encoder.
        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=1,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
            var_activation=None,
        )

        # y encoder: escape probability, input is s only
        self.y_encoder = Classifier(
            n_input=n_salient_latent,
            n_labels=1,
            n_layers=2,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=False,
            use_layer_norm=False,
            logits=True,
        )

        # Additive decoder: logits = linear(z) + nonlinear(s)
        self.decoder_z_out = nn.Linear(n_background_latent, n_input)

        self.decoder_s_fc = FCLayers(
            n_in=n_salient_latent,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
            use_layer_norm=False,
        )
        self.decoder_s_out = nn.Linear(n_hidden, n_input)

        # Zero-inflation logits
        self.px_dropout_decoder = nn.Linear(n_background_latent + n_hidden, n_input)

        self.gammas = torch.FloatTensor([10**x for x in range(-6, 7, 1)])

        if learn_basal_mean:
            self.basal_mean = torch.nn.Parameter(
                torch.randn(n_salient_latent), requires_grad=True
            )
        else:
            self.basal_mean = torch.nn.Parameter(
                torch.zeros(n_salient_latent), requires_grad=False
            )

        self.mmd_penalty = mmd_penalty
        self.inference_strategy = inference_strategy
        self.spatial_contrastive_weight = spatial_contrastive_weight
        self.spatial_repulsive_weight = spatial_repulsive_weight
        self.spatial_knn_k = spatial_knn_k

        # Spatial coordinate normalization buffers
        if spatial_mean is not None and spatial_std is not None:
            self.register_buffer("spatial_mean", torch.from_numpy(spatial_mean).float())
            self.register_buffer("spatial_std", torch.from_numpy(spatial_std).float())
        else:
            self.register_buffer("spatial_mean", torch.zeros(2))
            self.register_buffer("spatial_std", torch.ones(2))

        # Spatial prior network of z: XY -> prior mean of z (one network per section
        # if per_batch_prior > 0 and per_batch_prior_which includes "z")
        assert per_batch_prior_which in ("z", "s", "both"), per_batch_prior_which
        self.per_batch_prior = per_batch_prior
        self.umap_same_batch = umap_same_batch
        self.use_z_prior_net = use_z_prior_net
        self.umap_a = umap_a
        self.umap_b = umap_b
        self.prior_s_use_xy = prior_s_use_xy
        self.pbp_z = per_batch_prior > 0 and per_batch_prior_which in ("z", "both")
        self.pbp_s = per_batch_prior > 0 and per_batch_prior_which in ("s", "both")
        self._n_bg = n_background_latent
        self._n_sal = n_salient_latent
        _mk_z = lambda: Classifier(n_input=2, n_hidden=32, n_labels=n_background_latent,
                                   n_layers=1, logits=True)
        if self.pbp_z:
            self.spatial_prior_net = nn.ModuleList([_mk_z() for _ in range(per_batch_prior)])
        else:
            self.spatial_prior_net = _mk_z()

        # Perturbation prior network of s: one_hot(perturbation) + XY -> prior mean of s
        _mk_s = lambda: nn.Sequential(nn.Linear(n_labels + 2, 32), nn.ReLU(),
                                      nn.Linear(32, n_salient_latent))
        if self.pbp_s:
            self.perturbation_prior_net = nn.ModuleList(
                [_mk_s() for _ in range(per_batch_prior)])
        else:
            self.perturbation_prior_net = _mk_s()

        # Adversarial classifier on z
        self.adv_penalty = adv_penalty
        self.z_adversary = nn.Sequential(
            GradientReversal(lambda_=1.0),
            nn.Linear(n_background_latent, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_labels),
        )

        # Perturbation classifier on s
        self.cls_penalty = cls_penalty
        self.s_classifier = nn.Sequential(
            nn.Linear(n_salient_latent, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_labels),
        )

        self.kl_y_weight = kl_y_weight
        self.kl_z_weight = kl_z_weight
        self.kl_s_weight = kl_s_weight

    @auto_move_data
    def _compute_local_library_params(
        self, batch_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(
            one_hot(batch_index, n_batch), self.library_log_means
        )
        local_library_log_vars = F.linear(
            one_hot(batch_index, n_batch), self.library_log_vars
        )
        return local_library_log_means, local_library_log_vars

    @staticmethod
    def _get_min_batch_size(concat_tensors: Dict[str, Dict[str, torch.Tensor]]) -> int:
        return min(
            concat_tensors["background"][REGISTRY_KEYS.X_KEY].shape[0],
            concat_tensors["target"][REGISTRY_KEYS.X_KEY].shape[0],
        )

    @staticmethod
    def _reduce_tensors_to_min_batch_size(
        tensors: Dict[str, torch.Tensor], min_batch_size: int
    ) -> None:
        for name, tensor in tensors.items():
            tensors[name] = tensor[:min_batch_size, :]

    @staticmethod
    def _get_inference_input_from_concat_tensors(
        concat_tensors: Dict[str, Dict[str, torch.Tensor]], index: str
    ) -> Dict[str, torch.Tensor]:
        tensors = concat_tensors[index]
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        labels = tensors[REGISTRY_KEYS.LABELS_KEY]
        input_dict = dict(x=x, batch_index=batch_index, labels=labels)
        # Spatial coordinates (registered via continuous_covariate_keys)
        if REGISTRY_KEYS.CONT_COVS_KEY in tensors:
            input_dict["cont_covs"] = tensors[REGISTRY_KEYS.CONT_COVS_KEY]
        return input_dict

    def _get_inference_input(
        self, concat_tensors: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        background = self._get_inference_input_from_concat_tensors(
            concat_tensors, "background"
        )
        target = self._get_inference_input_from_concat_tensors(concat_tensors, "target")
        # Ensure batch sizes are the same.
        min_batch_size = self._get_min_batch_size(concat_tensors)
        self._reduce_tensors_to_min_batch_size(background, min_batch_size)
        self._reduce_tensors_to_min_batch_size(target, min_batch_size)
        return dict(background=background, target=target)

    @staticmethod
    def _get_generative_input_from_concat_tensors(
        concat_tensors: Dict[str, Dict[str, torch.Tensor]], index: str
    ) -> Dict[str, torch.Tensor]:
        tensors = concat_tensors[index]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        input_dict = dict(batch_index=batch_index)
        return input_dict

    @staticmethod
    def _get_generative_input_from_inference_outputs(
        inference_outputs: Dict[str, Dict[str, torch.Tensor]], data_source: str
    ) -> Dict[str, torch.Tensor]:
        z = inference_outputs[data_source]["z"]
        s = inference_outputs[data_source]["s"]
        library = inference_outputs[data_source]["library"]
        return dict(z=z, s=s, library=library)

    def _get_generative_input(
        self,
        concat_tensors: Dict[str, Dict[str, torch.Tensor]],
        inference_outputs: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        background_tensor_input = self._get_generative_input_from_concat_tensors(
            concat_tensors, "background"
        )
        target_tensor_input = self._get_generative_input_from_concat_tensors(
            concat_tensors, "target"
        )
        #Ensure batch sizes are the same.
        min_batch_size = self._get_min_batch_size(concat_tensors)
        self._reduce_tensors_to_min_batch_size(background_tensor_input, min_batch_size)
        self._reduce_tensors_to_min_batch_size(target_tensor_input, min_batch_size)

        background_inference_outputs = (
            self._get_generative_input_from_inference_outputs(
                inference_outputs, "background"
            )
        )
        target_inference_outputs = self._get_generative_input_from_inference_outputs(
            inference_outputs, "target"
        )
        background = {**background_tensor_input, **background_inference_outputs}
        target = {**target_tensor_input, **target_inference_outputs}
        return dict(background=background, target=target)

    @staticmethod
    def _reshape_tensor_for_samples(tensor: torch.Tensor, n_samples: int):
        return tensor.unsqueeze(0).expand((n_samples, tensor.size(0), tensor.size(1)))

    @auto_move_data
    def _generic_inference(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        labels: torch.Tensor,
        n_samples: int = 1,
        cont_covs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x_ = x
        if self.gene_likelihood == "normal":
            # Data is already log-normalized: no internal log1p, no library size
            # (Gaussian decoder ignores library). Encoder sees log-norm directly.
            library = torch.zeros((x.shape[0], 1), device=x.device)
        else:
            if self.use_observed_lib_size:
                library = torch.log(x.sum(1)).unsqueeze(1)
            x_ = torch.log(1 + x_)

        qz_m, qz_v, z = self.z_encoder(x_, batch_index)
        qs_m, qs_v, s = self.s_encoder(x_, batch_index)

        qy_logits = self.y_encoder(s)

        ql_m, ql_v = None, None
        if not self.use_observed_lib_size:
            ql_m, ql_v, library_encoded = self.l_encoder(x_, batch_index)
            library = library_encoded

        # Spatial conditional priors
        prior_z_m = None
        prior_s_m = None

        if cont_covs is not None:
            # cont_covs columns: X, Y and, if present, the section index (column 2).
            # They are used only by the prior networks and the spatial contrastive loss.
            xy = cont_covs[:, :2]
            bidx = (cont_covs[:, 2].long()
                    if (self.per_batch_prior > 0 and cont_covs.shape[1] >= 3) else None)
            cont_covs_normalized = (xy - self.spatial_mean) / (self.spatial_std + 1e-8)
            perturbation_onehot = F.one_hot(labels.long().squeeze(), num_classes=self.n_labels).float()
            if self.prior_s_use_xy:
                prior_s_input = torch.cat([perturbation_onehot, cont_covs_normalized], dim=-1)
            else:
                # Zero the coordinate inputs so that the prior of s depends on the label only.
                prior_s_input = torch.cat(
                    [perturbation_onehot, torch.zeros_like(cont_covs_normalized)], dim=-1)
            if bidx is None:
                prior_z_m = self.spatial_prior_net(cont_covs_normalized)
                prior_s_m = self.perturbation_prior_net(prior_s_input)
            else:
                def _per_batch(nets, inp, n_out):
                    out = inp.new_zeros((inp.shape[0], n_out))
                    for k in bidx.unique():
                        m = bidx == k
                        ki = int(min(max(int(k), 0), self.per_batch_prior - 1))
                        net = nets[ki]
                        if net.training and int(m.sum()) == 1:
                            # BatchNorm cannot run on a single cell in training mode.
                            net.eval(); out[m] = net(inp[m]); net.train()
                        else:
                            out[m] = net(inp[m])
                    return out
                prior_z_m = (_per_batch(self.spatial_prior_net, cont_covs_normalized, self._n_bg)
                             if self.pbp_z else self.spatial_prior_net(cont_covs_normalized))
                prior_s_m = (_per_batch(self.perturbation_prior_net, prior_s_input, self._n_sal)
                             if self.pbp_s else self.perturbation_prior_net(prior_s_input))

        if not self.use_z_prior_net:
            prior_z_m = None  # N(0, I) prior for z

        if n_samples > 1:
            qz_m = self._reshape_tensor_for_samples(qz_m, n_samples)
            qz_v = self._reshape_tensor_for_samples(qz_v, n_samples)
            z = self._reshape_tensor_for_samples(z, n_samples)
            qs_m = self._reshape_tensor_for_samples(qs_m, n_samples)
            qs_v = self._reshape_tensor_for_samples(qs_v, n_samples)
            s = self._reshape_tensor_for_samples(s, n_samples)

            if self.use_observed_lib_size:
                library = self._reshape_tensor_for_samples(library, n_samples)
            else:
                ql_m = self._reshape_tensor_for_samples(ql_m, n_samples)
                ql_v = self._reshape_tensor_for_samples(ql_v, n_samples)
                library = Normal(ql_m, ql_v.sqrt()).sample()

            if prior_s_m is not None:
                prior_s_m = self._reshape_tensor_for_samples(prior_s_m, n_samples)
            if prior_z_m is not None:
                prior_z_m = self._reshape_tensor_for_samples(prior_z_m, n_samples)

        outputs = dict(
            z=z,
            qz_m=qz_m,
            qz_v=qz_v,
            s=s,
            qs_m=qs_m,
            qs_v=qs_v,
            library=library,
            ql_m=ql_m,
            ql_v=ql_v,
            qy_logits=qy_logits,
            prior_s_m=prior_s_m,
            prior_z_m=prior_z_m,
        )
        return outputs

    @auto_move_data
    def inference(
        self,
        background: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        n_samples: int = 1,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        background_batch_size = background["x"].shape[0]
        target_batch_size = target["x"].shape[0]
        inference_input = {}
        for key in background.keys():
            inference_input[key] = torch.cat([background[key], target[key]], dim=0)
        outputs = self._generic_inference(**inference_input, n_samples=n_samples)

        batch_size_dim = 0 if n_samples == 1 else 1
        background_outputs, target_outputs = {}, {}
        for key in outputs.keys():
            if outputs[key] is not None:
                background_tensor, target_tensor = torch.split(
                    outputs[key],
                    [background_batch_size, target_batch_size],
                    dim=batch_size_dim,
                )
            else:
                background_tensor, target_tensor = None, None
            background_outputs[key] = background_tensor
            target_outputs[key] = target_tensor

        background_outputs["s"] = self.basal_mean.reshape(1, -1).repeat(
            background_outputs["s"].shape[0], 1
        )

        return dict(background=background_outputs, target=target_outputs)

    @auto_move_data
    def _generic_generative(
        self,
        z: torch.Tensor,
        s: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        h_s = self.decoder_s_fc(s)
        logits = self.decoder_z_out(z) + self.decoder_s_out(h_s)
        if self.gene_likelihood == "normal":
            # Additive decoder output IS the Gaussian mean (in log-norm space).
            # No softmax / no library / no dropout. px_r carries per-gene variance.
            px_mean = logits
            px_var = torch.exp(self.px_log_var)
            if self.px_var_max is not None:
                px_var = px_var.clamp(max=self.px_var_max)
            return dict(
                px_scale=px_mean, px_r=px_var, px_rate=px_mean, px_dropout=None
            )
        px_scale = F.softmax(logits, dim=-1)
        px_rate = torch.exp(library) * px_scale
        px_r = torch.exp(self.px_r)
        px_dropout = self.px_dropout_decoder(torch.cat([z, h_s], dim=-1))
        return dict(
            px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout
        )

    @auto_move_data
    def generative(
        self, background: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        latent_z_shape = background["z"].shape
        batch_size_dim = 0 if len(latent_z_shape) == 2 else 1
        background_batch_size = background["z"].shape[batch_size_dim]
        target_batch_size = target["z"].shape[batch_size_dim]

        # Background uses fixed basal_mean for s
        background_s_fixed = self.basal_mean.expand(background_batch_size, -1)
        if len(latent_z_shape) == 3:  # n_samples > 1
            n_samples = latent_z_shape[0]
            background_s_fixed = background_s_fixed.unsqueeze(0).expand(n_samples, -1, -1)

        generative_input = {}
        generative_input["z"] = torch.cat(
            [background["z"], target["z"]], dim=batch_size_dim
        )
        generative_input["s"] = torch.cat(
            [background_s_fixed, target["s"]], dim=batch_size_dim
        )
        generative_input["library"] = torch.cat(
            [background["library"], target["library"]], dim=batch_size_dim
        )
        generative_input["batch_index"] = torch.cat(
            [background["batch_index"], target["batch_index"]], dim=0
        )
        outputs = self._generic_generative(**generative_input)
        background_outputs, target_outputs = {}, {}
        for key in ["px_scale", "px_rate", "px_dropout"]:
            if outputs[key] is not None:
                background_tensor, target_tensor = torch.split(
                    outputs[key],
                    [background_batch_size, target_batch_size],
                    dim=batch_size_dim,
                )
            else:
                background_tensor, target_tensor = None, None
            background_outputs[key] = background_tensor
            target_outputs[key] = target_tensor
        background_outputs["px_r"] = outputs["px_r"]
        target_outputs["px_r"] = outputs["px_r"]
        return dict(background=background_outputs, target=target_outputs)

    def reconstruction_loss(
        self,
        x: torch.Tensor,
        px_rate: torch.Tensor,
        px_r: torch.Tensor,
        px_dropout: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the reconstruction (negative log-likelihood) loss.

        gene_likelihood == "zinb" (default): zero-inflated negative binomial on raw
        counts. px_rate = mean, px_r = inverse dispersion, px_dropout = zi logits.

        gene_likelihood == "normal": Gaussian on log-normalized data (scGen-style),
        for datasets without raw counts. px_rate = mean, px_r = per-gene variance,
        px_dropout unused.

        Returns
        -------
            Negative log likelihood for each data point, shape `(batch_size, )` or
            `(n_samples, batch_size)` if n_samples > 1.
        """
        if self.gene_likelihood == "normal":
            var = px_r  # per-gene variance vector, broadcast over cells
            recon_loss = (
                -Normal(px_rate, torch.sqrt(var + 1e-8)).log_prob(x).sum(dim=-1)
            )
            return recon_loss
        recon_loss = (
            -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
            .log_prob(x)
            .sum(dim=-1)
        )
        return recon_loss

    @staticmethod
    def latent_kl_divergence(
        variational_mean: torch.Tensor,
        variational_var: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between a variational posterior and prior Gaussian.
        Args:
        ----
            variational_mean: Mean of the variational posterior Gaussian.
            variational_var: Variance of the variational posterior Gaussian.
            prior_mean: Mean of the prior Gaussian.
            prior_var: Variance of the prior Gaussian.

        Returns
        -------
            KL divergence for each data point. If number of latent samples == 1,
            the tensor has shape `(batch_size, )`. If number of latent
            samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        return kl(
            Normal(variational_mean, variational_var.sqrt()),
            Normal(prior_mean, prior_var.sqrt()),
        ).sum(dim=-1)

    def library_kl_divergence(
        self,
        batch_index: torch.Tensor,
        variational_library_mean: torch.Tensor,
        variational_library_var: torch.Tensor,
        library: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between library size variational posterior and prior.

        Both the variational posterior and prior are Log-Normal.
        Args:
        ----
            batch_index: Batch indices for batch-specific library size mean and
                variance.
            variational_library_mean: Mean of variational Log-Normal.
            variational_library_var: Variance of variational Log-Normal.
            library: Sampled library size.

        Returns
        -------
            KL divergence for each data point. If number of latent samples == 1,
            the tensor has shape `(batch_size, )`. If number of latent
            samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        if not self.use_observed_lib_size:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)

            kl_library = kl(
                Normal(variational_library_mean, variational_library_var.sqrt()),
                Normal(local_library_log_means, local_library_log_vars.sqrt()),
            )
        else:
            kl_library = torch.zeros_like(library)
        return kl_library.sum(dim=-1)

    @torch.no_grad()
    def _compute_umap_p(
        self,
        dists: torch.Tensor,
        pair_ok: torch.Tensor,
        self_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute UMAP-style spatial similarity p_ij.

        For each cell i, finds sigma_i via binary search so that
        sum_j p_{j|i} = log2(spatial_knn_k).  Then symmetrizes.
        """
        N = dists.shape[0]
        device = dists.device
        target = math.log2(max(self.spatial_knn_k, 2))

        # Only allowed (same-perturbation, same-section if requested), non-self pairs
        valid = pair_ok & ~self_mask

        # rho_i = distance to nearest same-perturbation neighbor
        dists_masked = dists.masked_fill(~valid, float('inf'))
        rho = dists_masked.min(dim=1).values  # (N,)

        # d_ij - rho_i, clamped >= 0
        d_minus_rho = (dists - rho.unsqueeze(1)).clamp(min=0)  # (N, N)

        # Binary search for sigma_i
        lo = torch.full((N,), 1e-4, device=device)
        hi = torch.full((N,), 1e4, device=device)

        for _ in range(64):
            mid = (lo + hi) / 2
            p = torch.exp(-d_minus_rho / mid.unsqueeze(1))
            sum_p = (p * valid.float()).sum(dim=1)
            lo = torch.where(sum_p < target, mid, lo)
            hi = torch.where(sum_p >= target, mid, hi)

        sigma = (lo + hi) / 2

        # Final asymmetric p_{j|i}
        p_asym = torch.exp(-d_minus_rho / sigma.unsqueeze(1))
        p_asym = p_asym * valid.float()

        # Symmetrize: p_ij = p_{i|j} + p_{j|i} - p_{i|j} * p_{j|i}
        p_sym = p_asym + p_asym.T - p_asym * p_asym.T

        return p_sym

    def _spatial_contrastive_loss(
        self,
        s_features: torch.Tensor,
        labels: torch.Tensor,
        coords: torch.Tensor,
        batch_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """UMAP-style spatial contrastive loss (binary CE).

        Returns (attractive_loss, repulsive_loss) as separate scalars so they
        can be weighted independently.

        Attractive: -p_ij * log(q_ij)        — pull spatially close cells together in s
        Repulsive:  -(1-p_ij) * log(1-q_ij)  — push spatially distant cells apart in s

        p_ij: UMAP exponential kernel on physical coordinates (fixed, not learned)
        q_ij: Cauchy kernel on s-space distances (changes during training)
        """
        N = s_features.shape[0]
        device = s_features.device
        zero = torch.tensor(0.0, device=device)

        if N < 2:
            return zero, zero

        d = s_features.shape[1]

        # Pairwise spatial distances
        dists = torch.cdist(coords, coords)  # (N, N)
        self_mask = torch.eye(N, device=device, dtype=torch.bool)
        lf = labels.long().squeeze()
        pair_ok = lf.unsqueeze(0) == lf.unsqueeze(1)
        # With several sections, pair only cells of the same section.
        if batch_ids is not None:
            bf = batch_ids.long().squeeze()
            pair_ok = pair_ok & (bf.unsqueeze(0) == bf.unsqueeze(1))
        pair_mask = pair_ok & ~self_mask

        n_pairs = pair_mask.sum()
        if n_pairs == 0:
            return zero, zero

        # --- p_ij: spatial similarity (UMAP exponential kernel, no grad) ---
        p_ij = self._compute_umap_p(dists, pair_ok, self_mask)

        # --- q_ij: s-space similarity (Cauchy kernel) ---
        s_dists_norm = torch.cdist(s_features, s_features, p=2) / (d ** 0.5)
        if self.umap_a == 1.0 and self.umap_b == 1.0:
            q_ij = 1.0 / (1.0 + s_dists_norm ** 2)
        else:
            q_ij = 1.0 / (1.0 + self.umap_a * (s_dists_norm ** (2.0 * self.umap_b)))

        # --- Binary CE over legal pairs ---
        pair_float = pair_mask.float()
        n_pairs_f = n_pairs.float()

        att_num = -(p_ij * torch.log(q_ij + 1e-4) * pair_float).sum()
        rep_num = -((1.0 - p_ij) * torch.log(1.0 - q_ij + 1e-4) * pair_float).sum()

        return att_num / n_pairs_f, rep_num / n_pairs_f

    def _sum_with_free_bits(self, kl_perdim: torch.Tensor, free_bits: float) -> torch.Tensor:
        """Sum the per-dimension KL over dimensions; with free_bits > 0, dimensions whose
        KL averaged over the minibatch is below free_bits are not penalized.
        """
        if free_bits <= 0:
            return kl_perdim.sum(dim=-1)
        avg_dims = tuple(range(kl_perdim.dim() - 1))
        gate = (kl_perdim.mean(dim=avg_dims) > free_bits).to(kl_perdim.dtype)
        return (kl_perdim * gate).sum(dim=-1)

    def _generic_loss(
        self,
        tensors: Dict[str, torch.Tensor],
        inference_outputs: Dict[str, torch.Tensor],
        generative_outputs: Dict[str, torch.Tensor],
        is_target: bool = False,
    ) -> Dict[str, torch.Tensor]:
        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        qs_m = inference_outputs["qs_m"]
        qs_v = inference_outputs["qs_v"]
        s = inference_outputs["s"]
        qy_logits = inference_outputs["qy_logits"]

        if self.inference_strategy == "marginalize":
            qy_probs = F.sigmoid(qy_logits)
        else:
            qy_probs = gumbel_sigmoid(qy_logits, hard=True)

        library = inference_outputs["library"]
        ql_m = inference_outputs["ql_m"]
        ql_v = inference_outputs["ql_v"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        prior_s_v = torch.ones_like(qs_v)

        recon_loss = self.reconstruction_loss(x, px_rate, px_r, px_dropout)

        # KL of z: spatial prior if coordinates are given, N(0, I) otherwise
        prior_z_m = inference_outputs.get("prior_z_m", None)
        if prior_z_m is None:
            prior_z_m = torch.zeros_like(qz_m)
        kl_z = self._sum_with_free_bits(
            kl(Normal(qz_m, qz_v.sqrt()), Normal(prior_z_m, torch.ones_like(qz_v))),
            self.free_bits_z,
        )

        # KL of s (mixture prior) and KL of y
        qs = Normal(qs_m, qs_v.sqrt())
        ps_zero = Normal(self.basal_mean, prior_s_v)
        prior_s_m = inference_outputs.get("prior_s_m", None)
        if prior_s_m is not None:
            ps_one_mean = prior_s_m
        else:
            ps_one_mean = torch.zeros_like(qs_m)

        ps_one = Normal(ps_one_mean, prior_s_v)

        kl_s = qs.log_prob(s) - (
            qy_probs * ps_one.log_prob(s) + (1 - qy_probs) * ps_zero.log_prob(s)
        )
        kl_s = kl_s.sum(dim=-1)

        # Prior of y: Bernoulli(0.5)
        kl_y = kl(
            Bernoulli(probs=qy_probs),
            Bernoulli(probs=torch.tensor(0.5, device=qy_probs.device).expand_as(qy_probs))
        ).sum(dim=-1)
        kl_library = self.library_kl_divergence(batch_index, ql_m, ql_v, library)
        return dict(
            recon_loss=recon_loss,
            kl_z=kl_z,
            kl_s=kl_s,
            kl_y=kl_y,
            kl_library=kl_library,
        )

    def loss(
        self,
        concat_tensors: Dict[str, Dict[str, torch.Tensor]],
        inference_outputs: Dict[str, Dict[str, torch.Tensor]],
        generative_outputs: Dict[str, Dict[str, torch.Tensor]],
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """
        Compute loss terms for contrastive-VI.
        Args:
        ----
            concat_tensors: Tuple of data mini-batch. The first element contains
                background data mini-batch. The second element contains target data
                mini-batch.
            inference_outputs: Dictionary of inference step outputs. The keys
                are "background" and "target" for the corresponding outputs.
            generative_outputs: Dictionary of generative step outputs. The keys
                are "background" and "target" for the corresponding outputs.
            kl_weight: Importance weight for KL divergence of background and salient
                latent variables, relative to KL divergence of library size.

        Returns
        -------
            An scvi.module.base.LossOutput instance that records the following:
            loss: One-dimensional tensor for overall loss used for optimization.
            reconstruction_loss: Reconstruction loss with shape
                `(n_samples, batch_size)` if number of latent samples > 1, or
                `(batch_size, )` if number of latent samples == 1.
            kl_local: KL divergence term with shape
                `(n_samples, batch_size)` if number of latent samples > 1, or
                `(batch_size, )` if number of latent samples == 1.
            kl_global: One-dimensional tensor for global KL divergence term.
        """
        background_tensors = concat_tensors["background"]
        target_tensors = concat_tensors["target"]
        #Ensure batch sizes are the same.
        min_batch_size = self._get_min_batch_size(concat_tensors)
        self._reduce_tensors_to_min_batch_size(background_tensors, min_batch_size)
        self._reduce_tensors_to_min_batch_size(target_tensors, min_batch_size)

        background_losses = self._generic_loss(
            background_tensors,
            inference_outputs["background"],
            generative_outputs["background"],
        )
        target_losses = self._generic_loss(
            target_tensors,
            inference_outputs["target"],
            generative_outputs["target"],
            is_target=True,
        )
        reconst_loss = background_losses["recon_loss"] + target_losses["recon_loss"]
        kl_divergence_z = background_losses["kl_z"] + target_losses["kl_z"]
        kl_divergence_y = target_losses["kl_y"]
        kl_divergence_s = target_losses["kl_s"] + kl(
            Normal(
                inference_outputs["background"]["qs_m"],
                inference_outputs["background"]["qs_v"].sqrt(),
            ),
            Normal(
                self.basal_mean,
                torch.ones_like(inference_outputs["background"]["qs_v"]),
            ),
        ).sum(dim=-1)
        kl_divergence_l = background_losses["kl_library"] + target_losses["kl_library"]

        kl_local_for_warmup = (
            self.kl_z_weight * kl_divergence_z
            + self.kl_s_weight * kl_divergence_s
            + self.kl_y_weight * kl_divergence_y
        )
        kl_local_no_warmup = kl_divergence_l
        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        mmd_loss = mmd(
            inference_outputs["background"]["z"],
            inference_outputs["target"]["z"],
            gammas=self.gammas,
            device=self.device,
        )

        # Adversarial loss on z
        target_z = inference_outputs["target"]["z"]
        target_labels = target_tensors[REGISTRY_KEYS.LABELS_KEY].long().squeeze()
        adv_logits = self.z_adversary(target_z)
        adv_loss = F.cross_entropy(adv_logits, target_labels)

        # Gate-weighted perturbation classification loss on s
        target_qs_m_for_cls = inference_outputs["target"]["qs_m"]
        cls_logits = self.s_classifier(target_qs_m_for_cls)
        y_weight = torch.sigmoid(inference_outputs["target"]["qy_logits"]).squeeze().detach()
        cls_loss_per_cell = F.cross_entropy(cls_logits, target_labels, reduction='none')
        cls_loss = (y_weight * cls_loss_per_cell).mean()

        # Section index of each cell: cont_covs column 2 if present, else the batch key.
        def _slice_ids(tns):
            cc = tns.get(REGISTRY_KEYS.CONT_COVS_KEY, None)
            if cc is not None and cc.shape[1] >= 3:
                return cc[:, 2]
            return tns[REGISTRY_KEYS.BATCH_KEY]

        # Spatial UMAP CE loss on target s (attractive + repulsive)
        if (self.spatial_contrastive_weight > 0 or self.spatial_repulsive_weight > 0) and REGISTRY_KEYS.CONT_COVS_KEY in target_tensors:
            target_coords = target_tensors[REGISTRY_KEYS.CONT_COVS_KEY][:, :2]
            spatial_attractive, spatial_repulsive = self._spatial_contrastive_loss(
                s_features=inference_outputs["target"]["qs_m"],
                labels=target_tensors[REGISTRY_KEYS.LABELS_KEY],
                coords=target_coords,
                batch_ids=(_slice_ids(target_tensors) if self.umap_same_batch else None),
            )
        else:
            spatial_attractive = torch.tensor(0.0, device=reconst_loss.device)
            spatial_repulsive = torch.tensor(0.0, device=reconst_loss.device)

        loss = (
            torch.mean(reconst_loss + weighted_kl_local)
            + self.mmd_penalty * mmd_loss
            + self.adv_penalty * adv_loss
            + self.cls_penalty * cls_loss
            + self.spatial_contrastive_weight * spatial_attractive
            + self.spatial_repulsive_weight * spatial_repulsive
        )

        # Diagnostics (every 500 steps)
        if self.training:
            self._diag_step = getattr(self, '_diag_step', 0) + 1
            if self._diag_step % 500 == 0:
                with torch.no_grad():
                    t_qs_m = inference_outputs["target"]["qs_m"]
                    t_qy = torch.sigmoid(inference_outputs["target"]["qy_logits"])
                    d_basal = ((t_qs_m - self.basal_mean) ** 2).sum(-1).mean()
                    t_prior = inference_outputs["target"].get("prior_s_m", None)
                    d_prior = ((t_qs_m - t_prior) ** 2).sum(-1).mean() if t_prior is not None else torch.tensor(0.0)
                    yf = t_qy.squeeze()
                    hy, ly = yf > 0.8, yf < 0.2
                    db = ((t_qs_m - self.basal_mean) ** 2).sum(-1)
                    hb = db[hy].mean() if hy.any() else torch.tensor(0.0)
                    lb = db[ly].mean() if ly.any() else torch.tensor(0.0)
                    if t_prior is not None:
                        dp = ((t_qs_m - t_prior) ** 2).sum(-1)
                        hp = dp[hy].mean() if hy.any() else torch.tensor(0.0)
                        lp = dp[ly].mean() if ly.any() else torch.tensor(0.0)
                    else:
                        hp = lp = torch.tensor(0.0)
                    aa = (adv_logits.argmax(-1) == target_labels).float().mean()
                    ca = (cls_logits.argmax(-1) == target_labels).float().mean()
                print(f"\n[Step {self._diag_step}]")
                print(f"  kl_z: {kl_divergence_z.mean():.2f}, kl_s: {kl_divergence_s.mean():.2f}, kl_y: {kl_divergence_y.mean():.4f}")
                print(f"  y: mean={t_qy.mean():.4f}, min={t_qy.min():.4f}, max={t_qy.max():.4f}")
                b_qy = torch.sigmoid(inference_outputs["background"]["qy_logits"])
                print(f"  y_bg={b_qy.mean():.4f}  gap={t_qy.mean().item()-b_qy.mean().item():+.4f}")
                print(f"  s->basal: {d_basal:.2f}, s->pert: {d_prior:.2f}, gap: {(d_basal - d_prior):.2f}")
                print(f"  high_y({hy.sum().item()}): basal={hb:.2f} prior={hp:.2f} | low_y({ly.sum().item()}): basal={lb:.2f} prior={lp:.2f}")
                print(f"  recon: {reconst_loss.mean():.2f}, adv: {adv_loss:.3f}({aa:.3f}), cls: {cls_loss:.3f}({ca:.3f})")
                if self.spatial_contrastive_weight > 0 or self.spatial_repulsive_weight > 0:
                    sa_val = spatial_attractive.item() if torch.is_tensor(spatial_attractive) else spatial_attractive
                    sr_val = spatial_repulsive.item() if torch.is_tensor(spatial_repulsive) else spatial_repulsive
                    print(f"  spatial(s): attract={sa_val:.4f}(w={self.spatial_contrastive_weight}) repulse={sr_val:.4f}(w={self.spatial_repulsive_weight})")

        kl_local = dict(
            kl_divergence_l=kl_divergence_l,
            kl_divergence_z=kl_divergence_z,
            kl_divergence_s=kl_divergence_s,
            kl_divergence_y=kl_divergence_y,
        )
        kl_global = torch.tensor(0.0, device=reconst_loss.device)

        # Record the auxiliary losses separately in the training history.
        return LossOutput(
            loss, reconst_loss, kl_local, kl_global,
            extra_metrics={
                "cls_loss": cls_loss.detach(),
                "adv_loss": adv_loss.detach(),
                "mmd_loss": mmd_loss.detach(),
            },
        )

    @torch.no_grad()
    def sample(self):
        raise NotImplementedError

    @torch.no_grad()
    @auto_move_data
    def marginal_ll(self):
        raise NotImplementedError


class SpatialPerturbVI(BaseContrastiveModelClass):
    """
    Model class for contrastive-VI.
    Args:
    ----
        adata: AnnData object that has been registered via
            `ContrastiveVIModel.setup_anndata`.
        n_hidden: Number of nodes per hidden layer.
        n_latent: Dimensionality of the latent space.
        n_layers: Number of hidden layers used for encoder and decoder NNs.
        dropout_rate: Dropout rate for neural networks.
        use_observed_lib_size: Use observed library size for RNA as scaling factor in
            mean of conditional distribution.
        disentangle: Whether to disentangle the salient and background latent variables.
        use_mmd: Whether to use the maximum mean discrepancy loss to force background
            latent variables to have the same distribution for background and target
            data.
        mmd_weight: Weight used for the MMD loss.
        gammas: Gamma parameters for the MMD loss.
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 256,
        n_background_latent: int = 10,
        n_salient_latent: int = 10,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        use_observed_lib_size: bool = True,
        learn_basal_mean: bool = True,
        mmd_penalty: float = 0,
        adv_penalty: float = 5.0,
        cls_penalty: float = 1.0,
        kl_y_weight: float = 1.0,
        inference_strategy: Literal["marginalize", "gumbel_sigmoid"] = "marginalize",
        spatial_contrastive_weight: float = 0.0,
        spatial_repulsive_weight: float = 0.0,
        spatial_knn_k: int = 5,
        gene_likelihood: Literal["zinb", "normal"] = "zinb",
        learnable_px_var: bool = True,
        free_bits: float = 0.0,
        free_bits_z: Optional[float] = None,
        px_var_max: Optional[float] = None,
        kl_z_weight: float = 1.0,
        kl_s_weight: float = 1.0,
        per_batch_prior: int = 0,
        per_batch_prior_which: str = "both",
        umap_same_batch: bool = False,
        use_z_prior_net: bool = True,
        umap_a: float = 1.0,
        umap_b: float = 1.0,
        prior_s_use_xy: bool = True,
    ) -> None:
        super(SpatialPerturbVI, self).__init__(adata)

        n_batch = self.summary_stats.n_batch
        n_labels = self.summary_stats.n_labels
        use_size_factor_key = (
            REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        )
        library_log_means, library_log_vars = None, None
        if not use_size_factor_key:
            library_log_means, library_log_vars = _init_library_size(
                self.adata_manager, n_batch
            )

        spatial_mean = adata.uns.get("spatial_mean", None)
        spatial_std = adata.uns.get("spatial_std", None)

        self.module = ContrastiveVIPlusModule(
            n_input=self.summary_stats["n_vars"],
            n_batch=n_batch,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_background_latent=n_background_latent,
            n_salient_latent=n_salient_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_observed_lib_size=use_observed_lib_size,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            learn_basal_mean=learn_basal_mean,
            mmd_penalty=mmd_penalty,
            adv_penalty=adv_penalty,
            cls_penalty=cls_penalty,
            kl_y_weight=kl_y_weight,
            inference_strategy=inference_strategy,
            spatial_mean=spatial_mean,
            spatial_std=spatial_std,
            spatial_contrastive_weight=spatial_contrastive_weight,
            spatial_repulsive_weight=spatial_repulsive_weight,
            spatial_knn_k=spatial_knn_k,
            gene_likelihood=gene_likelihood,
            learnable_px_var=learnable_px_var,
            free_bits=free_bits,
            free_bits_z=free_bits_z,
            px_var_max=px_var_max,
            kl_z_weight=kl_z_weight,
            kl_s_weight=kl_s_weight,
            per_batch_prior=per_batch_prior,
            per_batch_prior_which=per_batch_prior_which,
            umap_same_batch=umap_same_batch,
            use_z_prior_net=use_z_prior_net,
            umap_a=umap_a,
            umap_b=umap_b,
            prior_s_use_xy=prior_s_use_xy,
        )
        self._model_summary_string = "Contrastive-VI."
        # Necessary line to get params to be used for saving and loading.
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        is_count_data: bool = True,
        **kwargs,
    ):
        """
        Set up AnnData instance for contrastive-VI model.

        Args:
        ----
            adata: AnnData object containing raw counts. Rows represent cells, columns
                represent features.
            layer: If not None, uses this as the key in adata.layers for raw count data.
            batch_key: Key in `adata.obs` for batch information. Categories will
                automatically be converted into integer categories and saved to
                `adata.obs["_scvi_batch"]`. If None, assign the same batch to all the
                data.
            labels_key: Key in `adata.obs` for label information. Categories will
                automatically be converted into integer categories and saved to
                `adata.obs["_scvi_labels"]`. If None, assign the same label to all the
                data.
            size_factor_key: Key in `adata.obs` for size factor information. Instead of
                using library size as a size factor, the provided size factor column
                will be used as offset in the mean of the likelihood. Assumed to be on
                linear scale.
            categorical_covariate_keys: Keys in `adata.obs` corresponding to categorical
                data. Used in some models.
            continuous_covariate_keys: Keys in `adata.obs` corresponding to continuous
                data. Used in some models.

        Returns
        -------
            If `copy` is True, return the modified `adata` set up for contrastive-VI
            model, otherwise `adata` is modified in place.
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=is_count_data),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        batch_size: Optional[int] = None,
        representation_kind: str = "salient",
    ) -> np.ndarray:
        """
        Return the background or salient latent representation for each cell.

        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices: Indices of cells in adata to use. If `None`, all cells are used.
        give_mean: Give mean of distribution or sample from it.
        batch_size: Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        representation_kind: Either "background" or "salient" for the corresponding
            representation kind.

        Returns
        -------
            A numpy array with shape `(n_cells, n_latent)`.
        """
        available_representation_kinds = ["background", "salient"]
        assert representation_kind in available_representation_kinds, (
            f"representation_kind = {representation_kind} is not one of"
            f" {available_representation_kinds}"
        )

        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )
        latent = []
        for tensors in data_loader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            labels = tensors[REGISTRY_KEYS.LABELS_KEY]
            cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None)
            outputs = self.module._generic_inference(
                x=x, batch_index=batch_index, labels=labels, n_samples=1, cont_covs=cont_covs
            )

            if representation_kind == "background":
                latent_m = outputs["qz_m"]
                latent_sample = outputs["z"]
            else:
                latent_m = outputs["qs_m"]
                latent_sample = outputs["s"]

            if give_mean:
                latent_sample = latent_m

            latent += [latent_sample.detach().cpu()]
        return torch.cat(latent).numpy()

    @torch.no_grad()
    def get_escape_probability(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return the escape probability (1 - perturbation success probability) for each cell.

        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices: Indices of cells in adata to use. If `None`, all cells are used.
        batch_size: Mini-batch size for data loading into model.

        Returns
        -------
            A numpy array with shape `(n_cells,)` containing escape probabilities.
            Values close to 0 indicate successful perturbation.
            Values close to 1 indicate escaped (not perturbed) cells.
        """
        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )
        probs = []
        for tensors in data_loader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            labels = tensors[REGISTRY_KEYS.LABELS_KEY]
            cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None)
            outputs = self.module._generic_inference(
                x=x, batch_index=batch_index, labels=labels, n_samples=1, cont_covs=cont_covs
            )
            qy_logits = outputs["qy_logits"]
            qy_probs = F.sigmoid(qy_logits)
            probs += [qy_probs.detach().cpu()]
        return torch.cat(probs).squeeze().numpy()

    @torch.no_grad()
    def predict(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        batch_size: Optional[int] = None,
        representation_kind: str = "salient",
    ) -> np.ndarray:
        """
        Return the background or salient latent representation for each cell.

        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices: Indices of cells in adata to use. If `None`, all cells are used.
        give_mean: Give mean of distribution or sample from it.
        batch_size: Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        representation_kind: Either "background" or "salient" for the corresponding
            representation kind.

        Returns
        -------
            A numpy array with shape `(n_cells, n_latent)`.
        """
        available_representation_kinds = ["background", "salient"]
        assert representation_kind in available_representation_kinds, (
            f"representation_kind = {representation_kind} is not one of"
            f" {available_representation_kinds}"
        )

        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )
        probs = []
        for tensors in data_loader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            labels = tensors[REGISTRY_KEYS.LABELS_KEY]
            cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None)
            outputs = self.module._generic_inference(
                x=x, batch_index=batch_index, labels=labels, n_samples=1, cont_covs=cont_covs
            )
            qy_logits = self.module.y_encoder(outputs["qs_m"])
            """
            qy_logits = self.module.y_encoder(
                torch.cat(
                    [
                        outputs["qs_m"],
                        F.one_hot(
                            labels.to(self.module.device).long(), num_classes=self.module.n_labels
                        ).squeeze(1),
                    ],
                    dim=1,
                )
            )
            """
            qy_probs = F.sigmoid(qy_logits)

            probs += [qy_probs.detach().cpu()]
        return torch.cat(probs).numpy()

    def get_normalized_expression_fold_change(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        transform_batch: Optional[Sequence[Union[Number, str]]] = None,
        gene_list: Optional[Sequence[str]] = None,
        library_size: Union[float, str] = 1.0,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return the normalized (decoded) gene expression.

        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices: Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch: Batch to condition on. If transform_batch is:
            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list: Return frequencies of expression for a subset of genes. This can
            save memory when working with large datasets and few genes are of interest.
        library_size:  Scale the expression frequencies to a common library size. This
            allows gene expression levels to be interpreted on a common scale of
            relevant magnitude. If set to `"latent"`, use the latent library size.
        n_samples: Number of posterior samples to use for estimation.
        batch_size: Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.

        Returns
        -------
            If `n_samples` > 1, then the shape is `(samples, cells, genes)`. Otherwise,
            shape is `(cells, genes)`. Each element is fold change of salient normalized
            expression divided by background normalized expression.
        """
        exprs = self.get_normalized_expression(
            adata=adata,
            indices=indices,
            transform_batch=transform_batch,
            gene_list=gene_list,
            library_size=library_size,
            n_samples=n_samples,
            batch_size=batch_size,
            return_mean=False,
            return_numpy=True,
        )
        salient_exprs = exprs["salient"]
        background_exprs = exprs["background"]
        fold_change = salient_exprs / background_exprs
        return fold_change

    @torch.no_grad()
    def get_normalized_expression(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        transform_batch: Optional[Sequence[Union[Number, str]]] = None,
        gene_list: Optional[Sequence[str]] = None,
        library_size: Union[float, Literal["latent"]] = 1.0,
        n_samples: int = 1,
        n_samples_overall: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """
        Return the normalized (decoded) gene expression.

        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices: Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch: Batch to condition on. If transform_batch is:
            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list: Return frequencies of expression for a subset of genes. This can
            save memory when working with large datasets and few genes are of interest.
        library_size:  Scale the expression frequencies to a common library size. This
            allows gene expression levels to be interpreted on a common scale of
            relevant magnitude. If set to `"latent"`, use the latent library size.
        n_samples: Number of posterior samples to use for estimation.
        n_samples_overall: The number of random samples in `adata` to use.
        batch_size: Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        return_mean: Whether to return the mean of the samples.
        return_numpy: Return a `numpy.ndarray` instead of a `pandas.DataFrame`.
            DataFrame includes gene names as columns. If either `n_samples=1` or
            `return_mean=True`, defaults to `False`. Otherwise, it defaults to `True`.

        Returns
        -------
            A dictionary with keys "background" and "salient", with value as follows.
            If `n_samples` > 1 and `return_mean` is `False`, then the shape is
            `(samples, cells, genes)`. Otherwise, shape is `(cells, genes)`. In this
            case, return type is `pandas.DataFrame` unless `return_numpy` is `True`.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        data_loader = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )

        transform_batch = list(
            _get_batch_code_from_category(
                self.get_anndata_manager(adata, required=True), transform_batch
            )
        )

        if gene_list is None:
            gene_mask = [True] * adata.n_vars
        else:
            gene_mask = [
                True if gene in gene_list else False for gene in adata.var_names
            ]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and"
                    " return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if library_size == "latent":
            generative_output_key = "px_rate"
            scaling = 1.0
        else:
            generative_output_key = "px_scale"
            scaling = library_size

        background_exprs = []
        salient_exprs = []
        for tensors in data_loader:
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            labels = tensors[REGISTRY_KEYS.LABELS_KEY]
            cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None)
            background_per_batch_exprs = []
            salient_per_batch_exprs = []
            for batch in transform_batch:
                if batch is not None:
                    batch_index = torch.ones_like(batch_index) * batch
                inference_outputs = self.module._generic_inference(
                    x=x, batch_index=batch_index, n_samples=n_samples, labels=labels, cont_covs=cont_covs
                )
                z = inference_outputs["z"]
                s = inference_outputs["s"]
                library = inference_outputs["library"]
                background_generative_outputs = self.module._generic_generative(
                    z=z, s=torch.zeros_like(s), library=library, batch_index=batch_index
                )
                salient_generative_outputs = self.module._generic_generative(
                    z=z, s=s, library=library, batch_index=batch_index
                )
                background_outputs = self._preprocess_normalized_expression(
                    background_generative_outputs,
                    generative_output_key,
                    gene_mask,
                    scaling,
                )
                background_per_batch_exprs.append(background_outputs)
                salient_outputs = self._preprocess_normalized_expression(
                    salient_generative_outputs,
                    generative_output_key,
                    gene_mask,
                    scaling,
                )
                salient_per_batch_exprs.append(salient_outputs)

            background_per_batch_exprs = np.stack(
                background_per_batch_exprs
            )  # Shape is (len(transform_batch) x batch_size x n_var).
            salient_per_batch_exprs = np.stack(salient_per_batch_exprs)
            background_exprs += [background_per_batch_exprs.mean(0)]
            salient_exprs += [salient_per_batch_exprs.mean(0)]

        if n_samples > 1:
            # The -2 axis correspond to cells.
            background_exprs = np.concatenate(background_exprs, axis=-2)
            salient_exprs = np.concatenate(salient_exprs, axis=-2)
        else:
            background_exprs = np.concatenate(background_exprs, axis=0)
            salient_exprs = np.concatenate(salient_exprs, axis=0)
        if n_samples > 1 and return_mean:
            background_exprs = background_exprs.mean(0)
            salient_exprs = salient_exprs.mean(0)

        if return_numpy is None or return_numpy is False:
            genes = adata.var_names[gene_mask]
            samples = adata.obs_names[indices]
            background_exprs = pd.DataFrame(
                background_exprs, columns=genes, index=samples
            )
            salient_exprs = pd.DataFrame(salient_exprs, columns=genes, index=samples)
        return {"background": background_exprs, "salient": salient_exprs}

    @torch.no_grad()
    def get_salient_normalized_expression(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        transform_batch: Optional[Sequence[Union[Number, str]]] = None,
        gene_list: Optional[Sequence[str]] = None,
        library_size: Union[float, str] = 1.0,
        n_samples: int = 1,
        n_samples_overall: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Return the normalized (decoded) gene expression.

        Gene expressions are decoded from both the background and salient latent space.

        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices: Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch: Batch to condition on. If transform_batch is:
            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list: Return frequencies of expression for a subset of genes. This can
            save memory when working with large datasets and few genes are of interest.
        library_size:  Scale the expression frequencies to a common library size. This
            allows gene expression levels to be interpreted on a common scale of
            relevant magnitude. If set to `"latent"`, use the latent library size.
        n_samples: Number of posterior samples to use for estimation.
        n_samples_overall: The number of random samples in `adata` to use.
        batch_size: Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        return_mean: Whether to return the mean of the samples.
        return_numpy: Return a `numpy.ndarray` instead of a `pandas.DataFrame`.
            DataFrame includes gene names as columns. If either `n_samples=1` or
            `return_mean=True`, defaults to `False`. Otherwise, it defaults to `True`.

        Returns
        -------
            If `n_samples` > 1 and `return_mean` is `False`, then the shape is
            `(samples, cells, genes)`. Otherwise, shape is `(cells, genes)`. In this
            case, return type is `pandas.DataFrame` unless `return_numpy` is `True`.
        """
        exprs = self.get_normalized_expression(
            adata=adata,
            indices=indices,
            transform_batch=transform_batch,
            gene_list=gene_list,
            library_size=library_size,
            n_samples=n_samples,
            n_samples_overall=n_samples_overall,
            batch_size=batch_size,
            return_mean=return_mean,
            return_numpy=return_numpy,
        )
        return exprs["salient"]

    @torch.no_grad()
    def get_specific_normalized_expression(
        self,
        adata: Optional[AnnData],
        indices: Optional[Sequence[int]] = None,
        transform_batch: Optional[Sequence[Union[Number, str]]] = None,
        gene_list: Optional[Sequence[str]] = None,
        library_size: Union[float, str] = 1,
        n_samples: int = 1,
        n_samples_overall: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
        expression_type: Optional[str] = None,
        indices_to_return_salient: Optional[Sequence[int]] = None,
    ):
        """
        Return the normalized (decoded) gene expression.

        Gene expressions are decoded from either the background or salient latent space.
        One of `expression_type` or `indices_to_return_salient` should have an input
        argument.

        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices: Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch: Batch to condition on. If transform_batch is:
            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list: Return frequencies of expression for a subset of genes. This can
            save memory when working with large datasets and few genes are of interest.
        library_size:  Scale the expression frequencies to a common library size. This
            allows gene expression levels to be interpreted on a common scale of
            relevant magnitude. If set to `"latent"`, use the latent library size.
        n_samples: Number of posterior samples to use for estimation.
        n_samples_overall: The number of random samples in `adata` to use.
        batch_size: Mini-batch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        return_mean: Whether to return the mean of the samples.
        return_numpy: Return a `numpy.ndarray` instead of a `pandas.DataFrame`.
            DataFrame includes gene names as columns. If either `n_samples=1` or
            `return_mean=True`, defaults to `False`. Otherwise, it defaults to `True`.
        expression_type: One of {"salient", "background"} to specify the type of
            normalized expression to return.
        indices_to_return_salient: If `indices` is a subset of
            `indices_to_return_salient`, normalized expressions derived from background
            and salient latent embeddings are returned. If `indices` is not `None` and
            is not a subset of `indices_to_return_salient`, normalized expressions
            derived only from background latent embeddings are returned.

        Returns
        -------
            If `n_samples` > 1 and `return_mean` is `False`, then the shape is
            `(samples, cells, genes)`. Otherwise, shape is `(cells, genes)`. In this
            case, return type is `pandas.DataFrame` unless `return_numpy` is `True`.
        """
        is_expression_type_none = expression_type is None
        is_indices_to_return_salient_none = indices_to_return_salient is None
        if is_expression_type_none and is_indices_to_return_salient_none:
            raise ValueError(
                "Both expression_type and indices_to_return_salient are None! "
                "Exactly one of them needs to be supplied with an input argument."
            )
        elif (not is_expression_type_none) and (not is_indices_to_return_salient_none):
            raise ValueError(
                "Both expression_type and indices_to_return_salient have an input "
                "argument! Exactly one of them needs to be supplied with an input "
                "argument."
            )
        else:
            exprs = self.get_normalized_expression(
                adata=adata,
                indices=indices,
                transform_batch=transform_batch,
                gene_list=gene_list,
                library_size=library_size,
                n_samples=n_samples,
                n_samples_overall=n_samples_overall,
                batch_size=batch_size,
                return_mean=return_mean,
                return_numpy=return_numpy,
            )
            if not is_expression_type_none:
                return exprs[expression_type]
            else:
                if indices is None:
                    indices = np.arange(adata.n_obs)
                if set(indices).issubset(set(indices_to_return_salient)):
                    return exprs["salient"]
                else:
                    return exprs["background"]

    def differential_expression(
        self,
        adata: Optional[AnnData] = None,
        groupby: Optional[str] = None,
        group1: Optional[Iterable[str]] = None,
        group2: Optional[str] = None,
        idx1: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
        idx2: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
        mode: str = "change",
        delta: float = 0.25,
        batch_size: Optional[int] = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: Optional[Iterable[str]] = None,
        batchid2: Optional[Iterable[str]] = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        target_idx: Optional[Sequence[int]] = None,
        n_samples: int = 1,
        **kwargs,
    ) -> pd.DataFrame:
        r"""
        Perform differential expression analysis.

        Args:
        ----
        adata: AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        groupby: The key of the observations grouping to consider.
        group1: Subset of groups, e.g. ["g1", "g2", "g3"], to which comparison shall be
            restricted, or all groups in `groupby` (default).
        group2: If `None`, compare each group in `group1` to the union of the rest of
            the groups in `groupby`. If a group identifier, compare with respect to this
            group.
        idx1: `idx1` and `idx2` can be used as an alternative to the AnnData keys.
            Custom identifier for `group1` that can be of three sorts:
            (1) a boolean mask, (2) indices, or (3) a string. If it is a string, then
            it will query indices that verifies conditions on adata.obs, as described
            in `pandas.DataFrame.query()`. If `idx1` is not `None`, this option
            overrides `group1` and `group2`.
        idx2: Custom identifier for `group2` that has the same properties as `idx1`.
            By default, includes all cells not specified in `idx1`.
        mode: Method for differential expression. See
            https://docs.scvi-tools.org/en/0.14.1/user_guide/background/differential_expression.html
            for more details.
        delta: Specific case of region inducing differential expression. In this case,
            we suppose that R\[-delta, delta] does not induce differential expression
            (change model default case).
        batch_size: Mini-batch size for data loading into model. Defaults to
            scvi.settings.batch_size.
        all_stats: Concatenate count statistics (e.g., mean expression group 1) to DE
            results.
        batch_correction: Whether to correct for batch effects in DE inference.
        batchid1: Subset of categories from `batch_key` registered in `setup_anndata`,
            e.g. ["batch1", "batch2", "batch3"], for `group1`. Only used if
            `batch_correction` is `True`, and by default all categories are used.
        batchid2: Same as `batchid1` for `group2`. `batchid2` must either have null
            intersection with `batchid1`, or be exactly equal to `batchid1`. When the
            two sets are exactly equal, cells are compared by decoding on the same
            batch. When sets have null intersection, cells from `group1` and `group2`
            are decoded on each group in `group1` and `group2`, respectively.
        fdr_target: Tag features as DE based on posterior expected false discovery rate.
        silent: If `True`, disables the progress bar. Default: `False`.
        target_idx: If not `None`, a boolean or integer identifier should be used for
            cells in the contrastive target group. Normalized expression values derived
            from both salient and background latent embeddings are used when
            {group1, group2} is a subset of the target group, otherwise background
            normalized expression values are used.

        **kwargs: Keyword args for
            `scvi.model.base.DifferentialComputation.get_bayes_factors`.

        Returns
        -------
        Differential expression DataFrame.
        """
        adata = self._validate_anndata(adata)
        col_names = adata.var_names

        if target_idx is not None:
            target_idx = np.array(target_idx)
            if target_idx.dtype is np.dtype("bool"):
                assert (
                    len(target_idx) == adata.n_obs
                ), "target_idx mask must be the same length as adata!"
                target_idx = np.arange(adata.n_obs)[target_idx]
            model_fn = partial(
                self.get_specific_normalized_expression,
                return_numpy=True,
                n_samples=n_samples,
                batch_size=batch_size,
                expression_type=None,
                indices_to_return_salient=target_idx,
            )
        else:
            model_fn = partial(
                self.get_specific_normalized_expression,
                return_numpy=True,
                n_samples=n_samples,
                batch_size=batch_size,
                expression_type="salient",
                indices_to_return_salient=None,
            )

        result = _de_core(
            self.get_anndata_manager(adata, required=True),
            model_fn,
            groupby=groupby,
            group1=group1,
            group2=group2,
            idx1=idx1,
            idx2=idx2,
            all_stats=all_stats,
            all_stats_fn=scrna_raw_counts_properties,
            col_names=col_names,
            mode=mode,
            batchid1=batchid1,
            batchid2=batchid2,
            delta=delta,
            batch_correction=batch_correction,
            fdr=fdr_target,
            silent=silent,
            **kwargs,
        )
        return result

    @staticmethod
    @torch.no_grad()
    def _preprocess_normalized_expression(
        generative_outputs: Dict[str, torch.Tensor],
        generative_output_key: str,
        gene_mask: Union[list, slice],
        scaling: float,
    ) -> np.ndarray:
        output = generative_outputs[generative_output_key]
        output = output[..., gene_mask]
        output *= scaling
        output = output.cpu().numpy()
        return output

    @torch.no_grad()
    def get_latent_library_size(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Returns the latent library size for each cell.
        This is denoted as :math:`\ell_n` in the scVI paper.
        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`,
            defaults to the AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        give_mean
            Return the mean or a sample from the posterior distribution.
        batch_size
            Minibatch size for data loading into model. Defaults to
            `scvi.settings.batch_size`.
        """
        self._check_if_trained(warn=False)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        libraries = []
        for tensors in scdl:
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None)
            labels = tensors[REGISTRY_KEYS.LABELS_KEY]
            outputs = self.module._generic_inference(x=x, batch_index=batch_index, labels=labels, cont_covs=cont_covs)

            library = outputs["library"]
            if not give_mean:
                library = torch.exp(library)
            else:
                ql = (outputs["ql_m"], outputs["ql_v"])
                if ql is None:
                    raise RuntimeError(
                        "The module for this model does not compute the posterior"
                        "distribution for the library size. Set `give_mean` to False"
                        "to use the observed library size instead."
                    )
                library = torch.distributions.LogNormal(ql[0], ql[1]).mean
            libraries += [library.cpu()]
        return torch.cat(libraries).numpy()

    @classmethod
    def load(
        cls,
        checkpoint: Dict[str, Any],
        adata: Optional[AnnData] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
    ):
        """
        Loads the model from the specified directory.

        Parameters
        ----------
        dir_path : `str`
            Path to saved model.
        adata : `~anndata.AnnData`, optional (default: `None`)
            Annotated data matrix. Will call `cpa.CPA.setup_anndata`
            on the data after model restoration.
        use_gpu : `bool` or `str` or `int`, optional (default: `None`)
            Whether a GPU should be used. If `True`, will use GPU.

        Returns
        -------
        :class:`~scvi.core.models.CPA`
            Restored model from the specified directory.
        """
        # To make scvi happy we'll make a temp directory
        # with the file it expects
        dir_path = tempfile.mkdtemp()
        # with open(os.path.join(dir_path, "CPA_info.json"), "w") as f:
        #    json.dump(checkpoint["cpa_info"], f)

        # Do some state_dict surgery
        model_state_dict = {}
        # ["covars_classifiers", "perturbation_classifier", "adv_loss_drugs"]
        blacklist_prefixes: List[str] = []
        for k, v in checkpoint["state_dict"].items():
            k = k.removeprefix("module.")
            if not any([k.startswith(prefix) for prefix in blacklist_prefixes]):
                model_state_dict[k] = v

        tmp_state = {
            "model_state_dict": model_state_dict,
            "var_names": checkpoint["var_names"],
            "attr_dict": checkpoint["attr_dict"],
        }
        torch.save(tmp_state, os.path.join(dir_path, "model.pt"))

        # with open(os.path.join(dir_path, "CPA_info.json")) as f:
        #    total_dict = json.load(f)
        #
        #    cls.pert_encoder = total_dict["pert_encoder"]
        #    cls.covars_encoder = total_dict["covars_encoder"]
        #    cls.pert_smiles_map = total_dict.get("pert_smiles_map", None)

        model = super().load(dir_path, adata=adata, use_gpu=use_gpu)
        # model.epoch_history = pd.DataFrame(checkpoint["epoch_history"])
        # Can't get this attribute into the lightning callback so
        # we just set it here; if we're loading from a checkpoint
        # it's obviously true anyway
        model.is_trained = True

        return model
