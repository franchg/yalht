from typing import Any, Dict, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class MNISTDataModule(LightningDataModule):
    """MNIST DataModule.

    A DataModule implements 5 key methods:
        - prepare_data: download data (called on 1 GPU/TPU only)
        - setup: load data, set variables (called on every GPU/TPU)
        - train_dataloader: return train dataloader
        - val_dataloader: return validation dataloader
        - test_dataloader: return test dataloader
        - predict_dataloader: return predict dataloader (optional)

    Read the docs:
        https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: tuple = (55000, 5000, 10000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize MNIST DataModule.

        Args:
            data_dir: Path to the data directory.
            train_val_test_split: Tuple with train, val, test sizes.
            batch_size: Batch size for dataloaders.
            num_workers: Number of workers for dataloaders.
            pin_memory: Whether to pin memory in dataloaders.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.train_set: Optional[MNIST] = None
        self.valid_set: Optional[MNIST] = None
        self.test_set: Optional[MNIST] = None

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return 10

    def prepare_data(self) -> None:
        """Download MNIST data if needed.

        Do not use it to assign state (self.x = y).
        """
        MNIST(self.hparams.data_dir, train=True, download=True)
        MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data and split datasets.

        Set variables: `self.train_set`, `self.valid_set`, `self.test_set`.

        This method is called by lightning with both `trainer.fit()` and
        `trainer.test()`, so be careful not to execute things like random split
        twice!
        """
        # load and split datasets only if not loaded already
        if not self.train_set and not self.valid_set and not self.test_set:
            train_set = MNIST(
                self.hparams.data_dir,
                train=True,
                transform=self.transforms,
            )
            test_set = MNIST(
                self.hparams.data_dir,
                train=False,
                transform=self.transforms,
            )
            dataset = ConcatDataset(datasets=[train_set, test_set])
            self.train_set, self.valid_set, self.test_set = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader."""
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader."""
        return DataLoader(
            dataset=self.valid_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader."""
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after fit or test."""
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Things to do when loading checkpoint."""
        pass
