"""DataModule for seismic dataset loading and augmentation in ASTF-net."""

import logging
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from astfnet.data_io.dataset import (
    SeismicDatasetHDF5,
)

logger = logging.getLogger(__name__)


class SeismicDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for seismic data loading and processing.

    It handles loading of seismic data from HDF5 files and provides DataLoader
    instances for training, validation, and testing.

    ``test_hdf5_files`` accepts a single path or a list of paths.  When a
    single string is provided it is normalised to a one-element list.  Use
    :meth:`set_test_file` to select which file is active before calling
    ``trainer.test()``, and :meth:`get_test_files` to retrieve the full list.
    """

    def __init__(
        self,
        train_hdf5_file: str,
        val_hdf5_file: str,
        test_hdf5_files: Optional[Union[str, List[str]]] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        log_normalize_astf: bool = True,
        log_normalize_input: bool = True,
        augmentation_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the SeismicDataModule.

        Args:
            train_hdf5_file: Path to training HDF5 file.
            val_hdf5_file: Path to validation HDF5 file.
            test_hdf5_files: Path(s) to test HDF5 file(s). A single string is
                normalised to a one-element list.
            batch_size: Batch size for all dataloaders.
            num_workers: Number of dataloader workers.
            log_normalize_astf: Apply log-normalisation to ASTF targets.
            log_normalize_input: Apply log-normalisation to inputs.
            augmentation_params: Dict with ``"augmentations"`` and
                ``"max_augmentations"`` keys. ``None`` disables augmentation.
        """
        super().__init__()
        self.train_hdf5_file = train_hdf5_file
        self.val_hdf5_file = val_hdf5_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.log_normalize_astf = log_normalize_astf
        self.log_normalize_input = log_normalize_input
        self.augmentation_params = augmentation_params or {
            "augmentations": [],
            "max_augmentations": 0,
        }

        # Normalise test files to a list
        if test_hdf5_files is None:
            self._test_hdf5_files: List[str] = []
        elif isinstance(test_hdf5_files, str):
            self._test_hdf5_files = [test_hdf5_files]
        else:
            self._test_hdf5_files = list(test_hdf5_files)

        # The file used for the *current* test run
        self._active_test_file: Optional[str] = self._test_hdf5_files[0] if self._test_hdf5_files else None

        self.dataset_class = SeismicDatasetHDF5

    # ------------------------------------------------------------------
    # Public helpers for multi-test-set evaluation
    # ------------------------------------------------------------------

    def get_test_files(self) -> List[str]:
        """Return the list of all configured test HDF5 file paths."""
        return list(self._test_hdf5_files)

    def set_test_file(self, test_file: str) -> None:
        """Select which test HDF5 file to use for the next ``trainer.test()``."""
        self._active_test_file = test_file

    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up training, validation, or test datasets based on the stage.

        Args:
            stage: One of {"fit", "validate", "test"} or None.
        """
        if stage == "fit" or stage is None:
            logger.info("Setting up training datasets...")
            self.train_dataset = self.dataset_class(
                self.train_hdf5_file,
                augmentation_params=self.augmentation_params,
                log_normalize_astf=self.log_normalize_astf,
                log_normalize_input=self.log_normalize_input,
            )
            logger.info("Setting up validation datasets...")

            self.val_dataset = self.dataset_class(
                self.val_hdf5_file,
                augmentation_params=None,
                log_normalize_astf=self.log_normalize_astf,
                log_normalize_input=self.log_normalize_input,
            )

        elif stage == "validate":
            logger.info("Setting up validation dataset...")
            self.val_dataset = self.dataset_class(
                self.val_hdf5_file,
                augmentation_params=None,
                log_normalize_astf=self.log_normalize_astf,
                log_normalize_input=self.log_normalize_input,
            )

        elif stage == "test":
            if self._active_test_file is None:
                raise ValueError(
                    "No test HDF5 file configured. Pass test_hdf5_files or call set_test_file() before testing."
                )
            logger.info("Setting up test dataset from: %s", self._active_test_file)
            self.test_dataset = self.dataset_class(
                self._active_test_file,
                augmentation_params=None,
                log_normalize_astf=self.log_normalize_astf,
                log_normalize_input=self.log_normalize_input,
            )

    def train_dataloader(self) -> DataLoader:
        """Create and return the training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
