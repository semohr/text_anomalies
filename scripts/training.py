import click
import sys
from typing import TypedDict
from ray.tune.integration.pytorch_lightning import TuneReportCallback

sys.path.append("../")
import pytorch_lightning as pl

import text_anomalies as ta


@click.command()
@click.option("--tune", default=False, help="Tune hyperparameters")
@click.option("--accelerator", default="gpu", help="Accelerator")
def main(max_epochs, min_epochs, tune, accelerator):
    """
    Trian the model on the DOEC dataset
    """

    # Tune hyperparameters
    if tune:
        # Find batch size
        # Find learning rate
        # Find latent size
        # Find hidden size
        # Find embedding size

        # Define search space
        search_space = {
            "batch_size": tune.choice([32, 64, 128]),
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "latent_size": tune.choice([16, 32, 64, 128, 256]),
            "hidden_size": tune.choice([128, 256, 512, 1024, 2048]),
            "embedding_size": tune.choice([128, 256, 512, 1024, 2048]),
            "num_layers": tune.choice([1, 2, 3, 4, 5]),
        }

        trainable = tune.with_parameters(
            train_model, data_dir="../data/doec", num_epochs=10
        )

        analysis = tune.run(
            trainable,
            resources_per_trial={"cpu": 4, "gpu": 1},
            metric="loss",
            mode="min",
            num_samples=10,
            config=search_space,
            name="tune_ssvae",
        )

        print(analysis.best_config)
    else:
        # Define config
        config = {
            "batch_size": 32,
            "learning_rate": 1e-3,
            "latent_size": 16,
            "hidden_size": 256,
            "embedding_size": 300,
            "num_layers": 2,
        }

        train_model(config, data_dir="../data/doec", num_epochs=-1)


class Cfg(TypedDict):
    batch_size: int
    learning_rate: float
    latent_size: int
    hidden_size: int
    embedding_size: int


def train_model(config: Cfg, data_dir="../data/doec", num_epochs=10):
    """Train a model using a given configuration. Can also
    be used for hyperparameter tuning."""

    model = ta.model.SSVAE(
        vocab_size=30_000,
        label_size=2,  # TODO
        embedding_size=config["embedding_size"],
        hidden_size=config["hidden_size"],
        latent_size=config["latent_size"],
        rnn_num_layers=config["num_layers"],
        learning_rate=config["learning_rate"],
    )

    data = ta.DOECDataModule(data_dir=data_dir)

    # Create trainer
    metrics = {
        "loss": "val_loss",
        "acc_labels": "val_acc_labels",
        "acc_seq": "val_acc_seq",
    }
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        progress_bar_refresh_rate=0,
        callbacks=[TuneReportCallback(metrics, on="validation_end")],
    )

    # Fit model
    trainer.fit(
        model,
        data,
    )
