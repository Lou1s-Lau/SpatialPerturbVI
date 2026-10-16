"""Generative model, inference networks and training objective of SpatialPerturbVI."""

import math
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scvi import REGISTRY_KEYS
from scvi.distributions import ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import Encoder, FCLayers
from torch import nn
from torch.distributions import Bernoulli, Normal
from torch.distributions import kl_divergence as kl

MMD_GAMMAS = [10.0**x for x in range(-6, 7)]


class MLP(nn.Module):
    """Fully connected hidden layers followed by a linear output layer."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int,
        n_layers: int,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            FCLayers(
                n_in=n_input,
                n_out=n_hidden,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                use_layer_norm=False,
                activation_fn=nn.ReLU,
            ),
            nn.Linear(n_hidden, n_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


class GradientReversal(nn.Module):
    """Identity in the forward pass, negated gradient in the backward pass."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversal.apply(x)


def _gram(x: torch.Tensor, y: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
    d2 = torch.square(torch.cdist(x, y, p=2.0))
    tmp = torch.matmul(gammas.unsqueeze(1), torch.reshape(d2, (1, -1)))
    return torch.reshape(torch.sum(torch.exp(-tmp), 0), d2.shape)


def mmd(x: torch.Tensor, y: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
    """Maximum mean discrepancy with a sum of Gaussian kernels."""
    cost = torch.mean(_gram(x, x, gammas))
    cost += torch.mean(_gram(y, y, gammas))
    cost -= 2 * torch.mean(_gram(x, y, gammas))
    if cost < 0:
        return cost.new_zeros(())
    return cost


class SpatialPerturbVIModule(BaseModuleClass):
    """SpatialPerturbVI.

    Each cell has a background latent ``z`` with a spatial prior
    ``N(g_z(coordinates), I)`` (one network per tissue section) and a salient latent
    ``s`` with a gated mixture prior: ``N(g_s(label, coordinates), I)`` with weight
    ``y`` and ``N(basal_mean, I)`` with weight ``1 - y``. Control cells are decoded
    with ``s = basal_mean``. The decoder is additive, ``f_z(z) + f_s(s)``.

    Parameters
    ----------
    n_input
        Number of genes.
    n_labels
        Number of perturbation labels, including the control label.
    n_sections
        Number of tissue sections (one prior network for ``z`` per section).
    n_hidden
        Width of the hidden layers.
    n_background_latent
        Dimension of ``z``.
    n_salient_latent
        Dimension of ``s``.
    n_layers
        Number of hidden layers of the encoders and of ``f_s``.
    dropout_rate
        Dropout rate.
    likelihood
        ``"zinb"`` for counts or ``"normal"`` for normalized expression.
    px_var_max
        Upper bound of the gene-wise variance of the Gaussian likelihood.
    kl_z_weight, kl_s_weight, kl_y_weight
        Weights of the KL terms of ``z``, ``s`` and ``y``.
    free_bits_z
        Dimensions of ``z`` whose KL, averaged over the minibatch, is below this
        value (nats) are not penalized. ``0`` disables it.
    cls_weight
        Weight of the gate-weighted perturbation classifier on ``s``.
    adv_weight
        Weight of the adversarial perturbation classifier on ``z``.
    mmd_weight
        Weight of the MMD between ``z`` of control and perturbed cells.
    spatial_attract_weight, spatial_repulse_weight
        Weights of the attractive and repulsive terms of the within-perturbation
        spatial contrastive loss on ``s``.
    spatial_knn_k
        The spatial affinity of a cell sums to ``log2(spatial_knn_k)``.
    pairs_within_section
        Restrict the spatial contrastive pairs to cells of the same section.
    spatial_mean, spatial_std
        Statistics used to standardize the coordinates for the prior networks.
    """

    def __init__(
        self,
        n_input: int,
        n_labels: int,
        n_sections: int = 1,
        n_hidden: int = 128,
        n_background_latent: int = 15,
        n_salient_latent: int = 48,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        likelihood: Literal["zinb", "normal"] = "zinb",
        px_var_max: Optional[float] = None,
        kl_z_weight: float = 1.0,
        kl_s_weight: float = 1.0,
        kl_y_weight: float = 1.0,
        free_bits_z: float = 0.0,
        cls_weight: float = 6.0,
        adv_weight: float = 5.0,
        mmd_weight: float = 1.0,
        spatial_attract_weight: float = 8.0,
        spatial_repulse_weight: float = 0.5,
        spatial_knn_k: int = 8,
        pairs_within_section: bool = False,
        spatial_mean: Optional[np.ndarray] = None,
        spatial_std: Optional[np.ndarray] = None,
    ):
        super().__init__()
        if likelihood not in ("zinb", "normal"):
            raise ValueError(f"Unknown likelihood: {likelihood}")
        self.n_input = n_input
        self.n_labels = n_labels
        self.n_sections = n_sections
        self.n_background_latent = n_background_latent
        self.n_salient_latent = n_salient_latent
        self.likelihood = likelihood
        self.px_var_max = px_var_max
        self.kl_z_weight = kl_z_weight
        self.kl_s_weight = kl_s_weight
        self.kl_y_weight = kl_y_weight
        self.free_bits_z = free_bits_z
        self.cls_weight = cls_weight
        self.adv_weight = adv_weight
        self.mmd_weight = mmd_weight
        self.spatial_attract_weight = spatial_attract_weight
        self.spatial_repulse_weight = spatial_repulse_weight
        self.spatial_knn_k = spatial_knn_k
        self.pairs_within_section = pairs_within_section

        if likelihood == "zinb":
            self.px_r = nn.Parameter(torch.randn(n_input))
        else:
            self.px_log_var = nn.Parameter(torch.zeros(n_input))

        encoder_kwargs = dict(
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution="normal",
            use_batch_norm=True,
            use_layer_norm=False,
            var_activation=None,
        )
        self.z_encoder = Encoder(n_input, n_background_latent, **encoder_kwargs)
        self.s_encoder = Encoder(n_input, n_salient_latent, **encoder_kwargs)
        self.gate_net = MLP(
            n_salient_latent, 1, n_hidden, n_layers=2, dropout_rate=dropout_rate,
            use_batch_norm=False,
        )

        self.decoder_z = nn.Linear(n_background_latent, n_input)
        self.decoder_s_hidden = FCLayers(
            n_in=n_salient_latent,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
            use_layer_norm=False,
        )
        self.decoder_s_out = nn.Linear(n_hidden, n_input)
        if likelihood == "zinb":
            self.dropout_decoder = nn.Linear(n_background_latent + n_hidden, n_input)

        self.basal_mean = nn.Parameter(torch.randn(n_salient_latent))

        if spatial_mean is None or spatial_std is None:
            spatial_mean, spatial_std = np.zeros(2), np.ones(2)
        self.register_buffer("spatial_mean", torch.as_tensor(spatial_mean).float())
        self.register_buffer("spatial_std", torch.as_tensor(spatial_std).float())

        self.z_prior_nets = nn.ModuleList(
            [MLP(2, n_background_latent, 32, n_layers=1) for _ in range(n_sections)]
        )
        self.s_prior_net = nn.Sequential(
            nn.Linear(n_labels + 2, 32), nn.ReLU(), nn.Linear(32, n_salient_latent)
        )
        self.adversary = nn.Sequential(
            GradientReversal(),
            nn.Linear(n_background_latent, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_labels),
        )
        self.perturbation_classifier = nn.Sequential(
            nn.Linear(n_salient_latent, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_labels)
        )
        self.gammas = torch.FloatTensor(MMD_GAMMAS)

    # ------------------------------------------------------------------ inference
    def _get_inference_input(self, tensors: Dict[str, Dict[str, torch.Tensor]]):
        return {
            group: dict(
                x=tensors[group][REGISTRY_KEYS.X_KEY],
                labels=tensors[group][REGISTRY_KEYS.LABELS_KEY],
                cont_covs=tensors[group][REGISTRY_KEYS.CONT_COVS_KEY],
            )
            for group in ("background", "target")
        }

    def _z_prior_mean(self, xy: torch.Tensor, section: Optional[torch.Tensor]) -> torch.Tensor:
        if self.n_sections == 1:
            return self.z_prior_nets[0](xy)
        out = xy.new_zeros((xy.shape[0], self.n_background_latent))
        for k in section.unique():
            mask = section == k
            net = self.z_prior_nets[int(min(max(int(k), 0), self.n_sections - 1))]
            if net.training and int(mask.sum()) == 1:
                # Batch normalization cannot be trained on a single cell.
                net.eval()
                out[mask] = net(xy[mask])
                net.train()
            else:
                out[mask] = net(xy[mask])
        return out

    @auto_move_data
    def _generic_inference(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        cont_covs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.likelihood == "zinb":
            library = torch.log(x.sum(1)).unsqueeze(1)
            x_ = torch.log(1 + x)
        else:
            library = torch.zeros((x.shape[0], 1), device=x.device)
            x_ = x

        qz_m, qz_v, z = self.z_encoder(x_)
        qs_m, qs_v, s = self.s_encoder(x_)
        qy_logits = self.gate_net(s)

        xy = (cont_covs[:, :2] - self.spatial_mean) / (self.spatial_std + 1e-8)
        section = cont_covs[:, 2].long() if self.n_sections > 1 else None
        onehot = F.one_hot(labels.long().squeeze(), num_classes=self.n_labels).float()
        prior_z_m = self._z_prior_mean(xy, section)
        prior_s_m = self.s_prior_net(torch.cat([onehot, xy], dim=-1))

        return dict(
            z=z, qz_m=qz_m, qz_v=qz_v,
            s=s, qs_m=qs_m, qs_v=qs_v,
            library=library, qy_logits=qy_logits,
            prior_z_m=prior_z_m, prior_s_m=prior_s_m,
        )

    @auto_move_data
    def inference(
        self, background: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        n_bg, n_tg = background["x"].shape[0], target["x"].shape[0]
        joint = {k: torch.cat([background[k], target[k]], dim=0) for k in background}
        outputs = self._generic_inference(**joint)
        bg_out, tg_out = {}, {}
        for k, v in outputs.items():
            bg_out[k], tg_out[k] = torch.split(v, [n_bg, n_tg], dim=0)
        bg_out["s"] = self.basal_mean.reshape(1, -1).repeat(n_bg, 1)
        return dict(background=bg_out, target=tg_out)

    # ----------------------------------------------------------------- generative
    def _get_generative_input(self, tensors, inference_outputs):
        return {
            group: {k: inference_outputs[group][k] for k in ("z", "s", "library")}
            for group in ("background", "target")
        }

    def _generic_generative(
        self, z: torch.Tensor, s: torch.Tensor, library: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        h_s = self.decoder_s_hidden(s)
        logits = self.decoder_z(z) + self.decoder_s_out(h_s)
        if self.likelihood == "normal":
            px_var = torch.exp(self.px_log_var)
            if self.px_var_max is not None:
                px_var = px_var.clamp(max=self.px_var_max)
            return dict(px_mean=logits, px_dropout=None, px_disp=px_var)
        px_scale = F.softmax(logits, dim=-1)
        px_rate = torch.exp(library) * px_scale
        px_dropout = self.dropout_decoder(torch.cat([z, h_s], dim=-1))
        return dict(
            px_scale=px_scale, px_mean=px_rate, px_dropout=px_dropout, px_disp=torch.exp(self.px_r)
        )

    @auto_move_data
    def generative(
        self, background: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        n_bg, n_tg = background["z"].shape[0], target["z"].shape[0]
        outputs = self._generic_generative(
            z=torch.cat([background["z"], target["z"]], dim=0),
            s=torch.cat([self.basal_mean.expand(n_bg, -1), target["s"]], dim=0),
            library=torch.cat([background["library"], target["library"]], dim=0),
        )
        bg_out, tg_out = {}, {}
        for k, v in outputs.items():
            if v is None or k == "px_disp":
                bg_out[k] = tg_out[k] = v
            else:
                bg_out[k], tg_out[k] = torch.split(v, [n_bg, n_tg], dim=0)
        return dict(background=bg_out, target=tg_out)

    @torch.no_grad()
    def decode_expected(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Expected expression for latents ``z`` and ``s``.

        For the ZINB likelihood, the expected proportion ``(1 - pi) * softmax(logits)``
        (multiply by a library size to obtain counts); for the Gaussian likelihood,
        the mean.
        """
        library = torch.zeros((z.shape[0], 1), device=z.device)
        out = self._generic_generative(z=z, s=s, library=library)
        if self.likelihood == "normal":
            return out["px_mean"]
        return (1 - torch.sigmoid(out["px_dropout"])) * out["px_scale"]

    # ----------------------------------------------------------------------- loss
    def _reconstruction_loss(self, x: torch.Tensor, gen: Dict[str, torch.Tensor]):
        if self.likelihood == "normal":
            scale = torch.sqrt(gen["px_disp"] + 1e-8)
            return -Normal(gen["px_mean"], scale).log_prob(x).sum(dim=-1)
        return (
            -ZeroInflatedNegativeBinomial(
                mu=gen["px_mean"], theta=gen["px_disp"], zi_logits=gen["px_dropout"]
            )
            .log_prob(x)
            .sum(dim=-1)
        )

    def _kl_z(self, inf: Dict[str, torch.Tensor]) -> torch.Tensor:
        kl_z = kl(
            Normal(inf["qz_m"], inf["qz_v"].sqrt()),
            Normal(inf["prior_z_m"], torch.ones_like(inf["qz_v"])),
        )
        if self.free_bits_z > 0:
            active = (kl_z.mean(dim=0) > self.free_bits_z).to(kl_z.dtype)
            kl_z = kl_z * active
        return kl_z.sum(dim=-1)

    @torch.no_grad()
    def _spatial_affinity(
        self, dists: torch.Tensor, pair_ok: torch.Tensor, self_mask: torch.Tensor
    ) -> torch.Tensor:
        """UMAP-style affinity p_ij between cells, computed within the minibatch."""
        n = dists.shape[0]
        target = math.log2(max(self.spatial_knn_k, 2))
        valid = pair_ok & ~self_mask
        rho = dists.masked_fill(~valid, float("inf")).min(dim=1).values
        d_minus_rho = (dists - rho.unsqueeze(1)).clamp(min=0)
        lo = torch.full((n,), 1e-4, device=dists.device)
        hi = torch.full((n,), 1e4, device=dists.device)
        for _ in range(64):
            mid = (lo + hi) / 2
            p = torch.exp(-d_minus_rho / mid.unsqueeze(1))
            sum_p = (p * valid.float()).sum(dim=1)
            lo = torch.where(sum_p < target, mid, lo)
            hi = torch.where(sum_p >= target, mid, hi)
        sigma = (lo + hi) / 2
        p_asym = torch.exp(-d_minus_rho / sigma.unsqueeze(1)) * valid.float()
        return p_asym + p_asym.T - p_asym * p_asym.T

    def _spatial_contrastive_loss(
        self,
        s: torch.Tensor,
        labels: torch.Tensor,
        coords: torch.Tensor,
        section: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attractive and repulsive cross-entropy between spatial and latent affinities.

        Pairs are perturbed cells of the minibatch that carry the same perturbation
        (and, optionally, lie in the same section).
        """
        n = s.shape[0]
        zero = torch.tensor(0.0, device=s.device)
        if n < 2:
            return zero, zero
        dists = torch.cdist(coords, coords)
        self_mask = torch.eye(n, device=s.device, dtype=torch.bool)
        lab = labels.long().squeeze()
        pair_ok = lab.unsqueeze(0) == lab.unsqueeze(1)
        if section is not None:
            sec = section.long().squeeze()
            pair_ok = pair_ok & (sec.unsqueeze(0) == sec.unsqueeze(1))
        pair_mask = pair_ok & ~self_mask
        n_pairs = pair_mask.sum()
        if n_pairs == 0:
            return zero, zero
        p_ij = self._spatial_affinity(dists, pair_ok, self_mask)
        s_dists = torch.cdist(s, s, p=2) / (s.shape[1] ** 0.5)
        q_ij = 1.0 / (1.0 + s_dists**2)
        pair_float = pair_mask.float()
        attract = -(p_ij * torch.log(q_ij + 1e-4) * pair_float).sum() / n_pairs.float()
        repulse = -((1.0 - p_ij) * torch.log(1.0 - q_ij + 1e-4) * pair_float).sum() / n_pairs.float()
        return attract, repulse

    def loss(
        self,
        tensors: Dict[str, Dict[str, torch.Tensor]],
        inference_outputs: Dict[str, Dict[str, torch.Tensor]],
        generative_outputs: Dict[str, Dict[str, torch.Tensor]],
        kl_weight: float = 1.0,
    ) -> LossOutput:
        bg_t, tg_t = tensors["background"], tensors["target"]
        bg_i, tg_i = inference_outputs["background"], inference_outputs["target"]
        bg_g, tg_g = generative_outputs["background"], generative_outputs["target"]

        reconst_loss = self._reconstruction_loss(
            bg_t[REGISTRY_KEYS.X_KEY], bg_g
        ) + self._reconstruction_loss(tg_t[REGISTRY_KEYS.X_KEY], tg_g)
        kl_z = self._kl_z(bg_i) + self._kl_z(tg_i)

        # Salient latent of perturbed cells: gated mixture prior.
        y = F.sigmoid(tg_i["qy_logits"])
        s = tg_i["s"]
        prior_v = torch.ones_like(tg_i["qs_v"])
        kl_s_target = Normal(tg_i["qs_m"], tg_i["qs_v"].sqrt()).log_prob(s) - (
            y * Normal(tg_i["prior_s_m"], prior_v).log_prob(s)
            + (1 - y) * Normal(self.basal_mean, prior_v).log_prob(s)
        )
        # Salient latent of control cells: basal component only.
        kl_s = kl_s_target.sum(dim=-1) + kl(
            Normal(bg_i["qs_m"], bg_i["qs_v"].sqrt()),
            Normal(self.basal_mean, torch.ones_like(bg_i["qs_v"])),
        ).sum(dim=-1)
        kl_y = kl(
            Bernoulli(probs=y),
            Bernoulli(probs=torch.tensor(0.5, device=y.device).expand_as(y)),
        ).sum(dim=-1)

        weighted_kl = kl_weight * (
            self.kl_z_weight * kl_z + self.kl_s_weight * kl_s + self.kl_y_weight * kl_y
        )

        mmd_loss = mmd(bg_i["z"], tg_i["z"], self.gammas.to(self.device))

        labels = tg_t[REGISTRY_KEYS.LABELS_KEY].long().squeeze()
        adv_loss = F.cross_entropy(self.adversary(tg_i["z"]), labels)

        cls_logits = self.perturbation_classifier(tg_i["qs_m"])
        gate = torch.sigmoid(tg_i["qy_logits"]).squeeze().detach()
        cls_loss = (gate * F.cross_entropy(cls_logits, labels, reduction="none")).mean()

        cont_covs = tg_t[REGISTRY_KEYS.CONT_COVS_KEY]
        attract, repulse = self._spatial_contrastive_loss(
            tg_i["qs_m"],
            tg_t[REGISTRY_KEYS.LABELS_KEY],
            cont_covs[:, :2],
            section=cont_covs[:, 2] if self.pairs_within_section else None,
        )

        loss = (
            torch.mean(reconst_loss + weighted_kl)
            + self.mmd_weight * mmd_loss
            + self.adv_weight * adv_loss
            + self.cls_weight * cls_loss
            + self.spatial_attract_weight * attract
            + self.spatial_repulse_weight * repulse
        )
        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=dict(kl_z=kl_z, kl_s=kl_s, kl_y=kl_y),
            extra_metrics=dict(
                cls_loss=cls_loss.detach(),
                adv_loss=adv_loss.detach(),
                mmd_loss=mmd_loss.detach(),
            ),
        )

    @torch.no_grad()
    def sample(self):
        raise NotImplementedError
