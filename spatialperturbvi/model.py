"""User-facing SpatialPerturbVI model."""

import math
from typing import Literal, Optional, Sequence

import numpy as np
import torch
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField, NumericalJointObsField
from scvi.dataloaders import AnnDataLoader
from scvi.model.base import BaseModelClass
from scvi.train import TrainingPlan, TrainRunner

from .data import ContrastiveDataSplitter
from .module import SpatialPerturbVIModule

SECTION_CODE_KEY = "_spatialperturbvi_section"


class SpatialPerturbVI(BaseModelClass):
    """Separates perturbation effects from tissue context in spatial CRISPR screens.

    Parameters
    ----------
    adata
        AnnData registered with :meth:`setup_anndata`.
    n_hidden, n_background_latent, n_salient_latent, n_layers, dropout_rate
        Network sizes.
    likelihood
        ``"zinb"`` for counts or ``"normal"`` for normalized expression.
    **module_kwargs
        Loss weights and other options of :class:`SpatialPerturbVIModule`.
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_background_latent: int = 15,
        n_salient_latent: int = 48,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        likelihood: Literal["zinb", "normal"] = "zinb",
        **module_kwargs,
    ):
        super().__init__(adata)
        args = self.adata_manager.registry["setup_args"]
        coords = adata.obs[[args["x_key"], args["y_key"]]].to_numpy()
        n_sections = 1
        if args["section_key"] is not None:
            n_sections = int(adata.obs[SECTION_CODE_KEY].max()) + 1
        self.module = SpatialPerturbVIModule(
            n_input=self.summary_stats.n_vars,
            n_labels=self.summary_stats.n_labels,
            n_sections=n_sections,
            n_hidden=n_hidden,
            n_background_latent=n_background_latent,
            n_salient_latent=n_salient_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            likelihood=likelihood,
            spatial_mean=coords.mean(axis=0),
            spatial_std=coords.std(axis=0),
            **module_kwargs,
        )
        self._model_summary_string = (
            f"SpatialPerturbVI: z dim {n_background_latent}, s dim {n_salient_latent}, "
            f"{likelihood} likelihood, {n_sections} section(s)"
        )
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        perturbation_key: str,
        control_label: str,
        x_key: str = "X",
        y_key: str = "Y",
        section_key: Optional[str] = None,
        layer: Optional[str] = None,
        is_count_data: bool = True,
        **kwargs,
    ):
        """Register an AnnData object.

        Parameters
        ----------
        adata
            Cells by genes. Counts for the ZINB likelihood, normalized expression for
            the Gaussian likelihood.
        perturbation_key
            Column of ``adata.obs`` with the perturbation of each cell.
        control_label
            Label of control cells in ``perturbation_key``.
        x_key, y_key
            Columns of ``adata.obs`` with the spatial coordinates.
        section_key
            Column of ``adata.obs`` with the tissue section of each cell, or ``None``
            for a single section. Coordinates are expected to be comparable only
            within a section.
        layer
            Layer with the expression values; ``None`` uses ``adata.X``.
        is_count_data
            Whether the expression values are counts.
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        cont_keys = [x_key, y_key]
        if section_key is not None:
            sections = adata.obs[section_key].astype(str).astype("category")
            adata.obs[SECTION_CODE_KEY] = sections.cat.codes.astype(float)
            cont_keys.append(SECTION_CODE_KEY)
        fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=is_count_data),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, perturbation_key),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, cont_keys),
        ]
        adata_manager = AnnDataManager(fields=fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def _group_indices(self, adata: AnnData):
        args = self.adata_manager.registry["setup_args"]
        is_control = (adata.obs[args["perturbation_key"]].astype(str) == args["control_label"]).to_numpy()
        return np.where(is_control)[0].tolist(), np.where(~is_control)[0].tolist()

    def train(
        self,
        max_epochs: int = 300,
        batch_size: int = 2048,
        lr: float = 2e-3,
        kl_warmup_steps: int = 120,
        train_size: float = 0.9,
        accelerator: str = "auto",
        devices="auto",
        **trainer_kwargs,
    ):
        """Train the model.

        Each step uses a minibatch of ``batch_size`` control cells and one of
        ``batch_size`` perturbed cells. The KL terms are warmed up over the first
        ``kl_warmup_steps`` gradient steps, rounded up to whole epochs.
        """
        background, target = self._group_indices(self.adata)
        splitter = ContrastiveDataSplitter(
            self.adata_manager, background, target, train_size=train_size, batch_size=batch_size
        )
        steps_per_epoch = max(
            splitter.n_background_train // batch_size, splitter.n_target_train // batch_size
        )
        warmup_epochs = max(1, math.ceil(kl_warmup_steps / steps_per_epoch))
        plan = TrainingPlan(self.module, lr=lr, n_epochs_kl_warmup=warmup_epochs)
        trainer_kwargs.setdefault("early_stopping", False)
        runner = TrainRunner(
            self,
            training_plan=plan,
            data_splitter=splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            **trainer_kwargs,
        )
        return runner()

    def _loader(self, adata, indices, batch_size):
        adata = self._validate_anndata(adata)
        return self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            shuffle=False,
            data_loader_class=AnnDataLoader,
        )

    def _inference(self, tensors):
        return self.module._generic_inference(
            x=tensors[REGISTRY_KEYS.X_KEY],
            labels=tensors[REGISTRY_KEYS.LABELS_KEY],
            cont_covs=tensors[REGISTRY_KEYS.CONT_COVS_KEY],
        )

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        kind: Literal["salient", "background"] = "salient",
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Posterior mean of the salient (``s``) or background (``z``) latent."""
        key = {"salient": "qs_m", "background": "qz_m"}[kind]
        self.module.eval()
        out = [self._inference(t)[key].cpu() for t in self._loader(adata, indices, batch_size)]
        return torch.cat(out).numpy()

    @torch.no_grad()
    def get_gate(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Gate value ``y``: probability that a cell is affected by its perturbation.

        Computed from samples of the salient latent and averaged over ``n_samples``.
        """
        self.module.eval()
        out = []
        for tensors in self._loader(adata, indices, batch_size):
            y = sum(
                torch.sigmoid(self._inference(tensors)["qy_logits"]) for _ in range(n_samples)
            )
            out.append((y / n_samples).cpu())
        return torch.cat(out).squeeze(-1).numpy()

    @torch.no_grad()
    def get_counterfactual_expression(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: int = 4096,
    ) -> np.ndarray:
        """Expression expected for each cell without its perturbation.

        Decoded from the cell's own background latent with the salient latent set to
        the basal state. For the ZINB likelihood, the expected proportion of each gene
        (multiply by a library size for counts); for the Gaussian likelihood, the mean.
        """
        z = self.get_latent_representation(adata, indices, kind="background")
        self.module.eval()
        device = self.module.device
        basal = self.module.basal_mean.detach().reshape(1, -1)
        out = []
        for i in range(0, len(z), batch_size):
            zb = torch.as_tensor(z[i : i + batch_size], device=device)
            out.append(self.module.decode_expected(zb, basal.expand(len(zb), -1)).cpu())
        return torch.cat(out).numpy()
