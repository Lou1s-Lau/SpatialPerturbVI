"""Paired minibatches of control (background) and perturbed (target) cells."""

from itertools import cycle
from typing import List, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from scvi import settings
from scvi.data import AnnDataManager
from scvi.dataloaders._concat_dataloader import ConcatDataLoader
from scvi.dataloaders._data_splitting import validate_data_split


class _PairedIterator:
    def __init__(self, background, target):
        self.background = iter(background)
        self.target = iter(target)

    def __iter__(self):
        return self

    def __next__(self):
        return {"background": next(self.background), "target": next(self.target)}


class ContrastiveDataLoader(ConcatDataLoader):
    """Yields ``{"background": batch, "target": batch}``.

    Iterates over the larger group and cycles through the smaller one.
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        background_indices: List[int],
        target_indices: List[int],
        shuffle: bool = False,
        batch_size: int = 128,
        drop_last: bool = False,
        **data_loader_kwargs,
    ):
        super().__init__(
            adata_manager=adata_manager,
            indices_list=[background_indices, target_indices],
            shuffle=shuffle,
            batch_size=batch_size,
            drop_last=drop_last,
            **data_loader_kwargs,
        )

    def __iter__(self):
        loaders = [dl if dl == self.largest_dl else cycle(dl) for dl in self.dataloaders]
        return _PairedIterator(background=loaders[0], target=loaders[1])


class ContrastiveDataSplitter(pl.LightningDataModule):
    """Splits control and perturbed cells separately into training and validation sets."""

    def __init__(
        self,
        adata_manager: AnnDataManager,
        background_indices: List[int],
        target_indices: List[int],
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
    ):
        super().__init__()
        self.adata_manager = adata_manager
        self.background_indices = background_indices
        self.target_indices = target_indices
        self.train_size = train_size
        self.validation_size = validation_size
        self.batch_size = batch_size
        self.n_background_train, self.n_background_val = validate_data_split(
            len(background_indices), train_size, validation_size
        )
        self.n_target_train, self.n_target_val = validate_data_split(
            len(target_indices), train_size, validation_size
        )
        if batch_size > min(self.n_background_train, self.n_target_train):
            raise ValueError(
                f"batch_size={batch_size} is larger than the smaller training group "
                f"({min(self.n_background_train, self.n_target_train)} cells)."
            )
        self.setup()

    def setup(self, stage: Optional[str] = None):
        random_state = np.random.RandomState(seed=settings.seed)
        tg = random_state.permutation(self.target_indices)
        self.target_val_idx = tg[: self.n_target_val]
        self.target_train_idx = tg[self.n_target_val : self.n_target_val + self.n_target_train]
        bg = random_state.permutation(self.background_indices)
        self.background_val_idx = bg[: self.n_background_val]
        self.background_train_idx = bg[
            self.n_background_val : self.n_background_val + self.n_background_train
        ]
        self.train_idx = np.concatenate((self.background_train_idx, self.target_train_idx))
        self.val_idx = np.concatenate((self.background_val_idx, self.target_val_idx))
        self.test_idx = np.array([], dtype=int)

    def _loader(self, background_indices, target_indices) -> ContrastiveDataLoader:
        return ContrastiveDataLoader(
            self.adata_manager,
            background_indices,
            target_indices,
            shuffle=True,
            drop_last=True,
            batch_size=self.batch_size,
            pin_memory=torch.cuda.is_available(),
        )

    def train_dataloader(self) -> ContrastiveDataLoader:
        return self._loader(self.background_train_idx, self.target_train_idx)

    def val_dataloader(self):
        if len(self.background_val_idx) > 0 and len(self.target_val_idx) > 0:
            return self._loader(self.background_val_idx, self.target_val_idx)
        return None
