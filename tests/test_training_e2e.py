from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import pytest
import pytorch_lightning as pl

from astfnet.data_io.datamodule import SeismicDataModule
from astfnet.models import ASTFModule
from astfnet.models.optimizer import OptimizerFactory
from astfnet.models.scheduler import SchedulerFactory

SEQ_LEN = 256
OUTPUT_LENGTH = 256


def _write_dummy_hdf5(path: Path, n_samples: int = 4) -> None:
    rng = np.random.default_rng(1234)
    with h5py.File(path, "w") as f:
        f.create_dataset("target_waveforms", data=rng.normal(size=(n_samples, SEQ_LEN)).astype(np.float32))
        f.create_dataset("egfs", data=rng.normal(size=(n_samples, SEQ_LEN)).astype(np.float32))
        f.create_dataset("astfs", data=np.abs(rng.normal(size=(n_samples, OUTPUT_LENGTH))).astype(np.float32))


def _base_train_config(model_name: str) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "loss": "mse",
        "batch_size": 2,
        "num_workers": 0,
        "log_normalize_astf": False,
        "log_normalize_input": True,
        "augmentations": [],
        "max_augmentations": 0,
        "max_epochs": 1,
        "device": "cpu",
        "gpus": 1,
        "in_channels": 2,
        "output_length": OUTPUT_LENGTH,
        "optimizer": {
            "name": "Adam",
            "lr": 1e-3,
        },
        "callbacks": {
            "lr_scheduler": {
                "name": "ReduceLROnPlateau",
                "monitor": "val/loss_epoch",
                "mode": "min",
                "factor": 0.5,
                "patience": 1,
            },
        },
    }


def _model_train_config(model_name: str) -> Dict[str, Any]:
    config = _base_train_config(model_name)
    model_configs: Dict[str, Dict[str, Any]] = {
        "simplecnn": {},
        "transformer": {
            "cnn_kernel_size": 16,
            "cnn_stride": 16,
            "d_model": 16,
            "nhead": 2,
            "num_encoder_layers": 1,
            "dim_feedforward": 32,
            "dropout": 0.0,
        },
        "unet1d": {
            "dropout_shallow": 0.0,
            "dropout_deep": 0.0,
        },
        "crdnn": {
            "cnn_channels": [8, 16],
            "cnn_kernel_size": 5,
            "cnn_pool_size": 2,
            "rnn_name": "GRU",
            "rnn_hidden_size": 16,
            "rnn_layers": 1,
            "rnn_bidirectional": True,
            "dnn_hidden_size": 32,
            "dnn_layers": 1,
            "dropout": 0.0,
        },
    }
    config.update(model_configs[model_name])
    return config


@pytest.mark.parametrize("model_name", ["simplecnn", "transformer", "unet1d", "crdnn"])
def test_train_py_style_fit_runs_one_training_step_for_implemented_models(tmp_path: Path, model_name: str) -> None:
    train_hdf5_file = tmp_path / "train.h5"
    val_hdf5_file = tmp_path / "val.h5"
    test_hdf5_file = tmp_path / "test.h5"
    for path in [train_hdf5_file, val_hdf5_file, test_hdf5_file]:
        _write_dummy_hdf5(path)

    config = _model_train_config(model_name)
    augmentation_params = {
        "augmentations": config.get("augmentations", config.get("data_augmentations", [])),
        "max_augmentations": int(config.get("max_augmentations", 0)),
    }
    datamodule = SeismicDataModule(
        train_hdf5_file=str(train_hdf5_file),
        val_hdf5_file=str(val_hdf5_file),
        test_hdf5_files=[str(test_hdf5_file)],
        batch_size=config.get("batch_size", 32),
        num_workers=config.get("num_workers", 2),
        log_normalize_astf=config.get("log_normalize_astf", True),
        log_normalize_input=config.get("log_normalize_input", True),
        augmentation_params=augmentation_params,
    )

    optimizer_factory = OptimizerFactory.from_config(config)
    scheduler_factory = SchedulerFactory.from_config(config)
    model = ASTFModule(config, optimizer_factory=optimizer_factory, scheduler_factory=scheduler_factory)

    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator=config["device"],
        devices=config["gpus"],
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=datamodule)

    assert trainer.global_step == 1
