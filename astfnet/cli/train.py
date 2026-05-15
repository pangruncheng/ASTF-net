import argparse
import logging
import os
import sys
from typing import Dict

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from astfnet.constants import resolve_data_paths
from astfnet.data_io.datamodule import SeismicDataModule
from astfnet.models import ASTFModule
from astfnet.models.optimizer import OptimizerFactory
from astfnet.models.scheduler import SchedulerFactory

logger = logging.getLogger(__name__)


def main() -> None:
    """Main function to train the ASTF-net model."""
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train ASTF-net with PyTorch Lightning.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data directory",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        default=False,
        help="Skip test-set evaluation after training",
    )
    args = parser.parse_args()

    # Load config
    config: Dict = dict(OmegaConf.load(args.config))

    # Resolve canonical data file paths from --data CLI argument
    data_path = args.data
    resolved = resolve_data_paths(data_path)
    train_dhf5_file = resolved["train_hdf5_file"]
    val_hdf5_file = resolved["val_hdf5_file"]
    test_hdf5_files = resolved["test_hdf5_files"]

    max_epochs = config["max_epochs"]

    augmentation_params = {
        "augmentations": config.get("augmentations", config.get("data_augmentations", [])),
        "max_augmentations": int(config.get("max_augmentations", 0)),
    }

    datamodule = SeismicDataModule(
        train_hdf5_file=train_dhf5_file,
        val_hdf5_file=val_hdf5_file,
        test_hdf5_files=test_hdf5_files,
        batch_size=config.get("batch_size", 32),
        num_workers=config.get("num_workers", 2),
        log_normalize_astf=config.get("log_normalize_astf", True),
        log_normalize_input=config.get("log_normalize_input", True),
        augmentation_params=augmentation_params,
    )

    # Build optimizer and scheduler factories from config
    optimizer_factory = OptimizerFactory.from_config(config)
    scheduler_factory = SchedulerFactory.from_config(config)

    # Model (auto-selected by config["model_name"])
    scheduler_name = scheduler_factory.name if scheduler_factory is not None else "none"
    logger.info(
        f"Instantiating model {config['model_name']} with optimizer {optimizer_factory.name} and scheduler {scheduler_name}..."
    )
    model = ASTFModule(config, optimizer_factory=optimizer_factory, scheduler_factory=scheduler_factory)

    device = config["device"]
    gpus = config["gpus"]

    tb_logger = TensorBoardLogger(save_dir=config["tb_output_dir"], name=config["tb_exp_name"])

    # --- Callbacks ---
    early_stop_callback = EarlyStopping(
        monitor=config["callbacks"]["early_stopping"]["monitor"],
        patience=config["callbacks"]["early_stopping"]["patience"],
        mode=config["callbacks"]["early_stopping"]["mode"],
        min_delta=config["callbacks"]["early_stopping"]["min_delta"],
        verbose=config["callbacks"]["early_stopping"]["verbose"],
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")  # Log LR every step

    checkpoint_callback = ModelCheckpoint(
        monitor=config["callbacks"]["model_checkpoint"]["monitor"],
        mode=config["callbacks"]["model_checkpoint"]["mode"],
        save_top_k=config["callbacks"]["model_checkpoint"]["save_top_k"],
        filename=config["callbacks"]["model_checkpoint"]["filename"],
        save_last=True,  # Save the last checkpoint as well
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=device,
        devices=gpus,
        default_root_dir="outputs",
        log_every_n_steps=10,
        logger=tb_logger,
        callbacks=[early_stop_callback, lr_monitor, checkpoint_callback],
    )

    logger.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)

    # --- Test-set evaluation ---
    skip_test = args.skip_test
    if not skip_test:
        logger.info("Starting evaluation on test sets...")
        test_files = datamodule.get_test_files()
        if not test_files:
            logger.info(" No test files configured – skipping test evaluation.")
        else:
            for test_file in test_files:
                test_name = os.path.splitext(os.path.basename(test_file))[0]
                logger.info(f"Evaluating on test set: {test_name}")

                model.set_test_prefix(test_name)
                datamodule.set_test_file(test_file)
                trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
