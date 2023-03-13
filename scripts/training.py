import click
from typing import TypedDict
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import text_anomalies as ta
import torch

    

@click.command()
@click.option("--tunning", is_flag=True, default=False, help="Tune hyperparameters")
@click.option("--accelerator", default="gpu", help="Accelerator")
def main(tunning=False, accelerator="gpu"):
    """
    Trian the model on the DOEC dataset
    """

    # Tune hyperparameters
    if tunning:
        # Find batch size
        # Find learning rate
        # Find latent size
        # Find hidden size
        # Find embedding size

        # Define search space
        search_space = {
            "batch_size": tune.choice([92]),
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "latent_size": tune.choice([20]),
            "hidden_size": tune.choice([512, ]),
            "embedding_size": tune.choice([300,]),
            "num_layers": tune.choice([2, 4]),
            "nu1" : tune.uniform(0.05, 0.2),
            "nu2" : tune.uniform(5, 100),
        }

        trainable = tune.with_parameters(
            train_model, data_dir="/home/smohr/Repositories/text_anomalies/data/doec/", num_epochs=5, tunning=True
        )

        analysis = tune.run(
            trainable,
            resources_per_trial={"cpu": 4, "gpu": 1},
            metric="loss",
            mode="min",
            num_samples=12,
            config=search_space,
            name="tune_ssvae_v2",
        )

        print(analysis.best_config)
    else:
        # Define config
        config = {
            "batch_size": 92,
            "learning_rate": 1e-4,
            "latent_size": 65,
            "hidden_size": 512,
            "embedding_size": 300,
            "num_layers": 5,
            "nu1" : 0.3,
            "nu2" : 0.5,
        }

        train_model(config, data_dir="/home/smohr/Repositories/text_anomalies/data/doec/", num_epochs=-1)


class Cfg(TypedDict):
    batch_size: int
    learning_rate: float
    latent_size: int
    hidden_size: int
    embedding_size: int
    nu1: float
    nu2: float


def train_model(config: Cfg, data_dir="/home/smohr/Repositories/text_anomalies/data/doec/", num_epochs=10, tunning=False):
    """Train a model using a given configuration. Can also
    be used for hyperparameter tuning."""
    torch.set_float32_matmul_precision('medium')
    data = ta.DOECDataModule(data_dir=data_dir, batch_size=config["batch_size"])
    data.prepare_data()
    data.setup()

    # Reduces size if tunning
    model = ta.model.SSVAE(
        vocab_size=30_000,
        label_size=data.num_classes,
        embedding_size=config["embedding_size"],
        hidden_size=config["hidden_size"],
        latent_size=config["latent_size"],
        rnn_num_layers=config["num_layers"],
        learning_rate=config["learning_rate"],
        nu1=config["nu1"],
        nu2=config["nu2"],
    )


    # Create trainer
    metrics = {
        "loss": "val_loss",
        "acc_labels": "val_acc_labels",
        "acc_seq": "val_acc_seq",
    }

    extra = {}
    if tunning:
        callbacks = [TuneReportCallback(metrics, on="validation_end")]
        extra = {"enable_progress_bar": False, "callbacks": callbacks}
    else:
        callbacks = [EarlyStopping(monitor="val_loss",
                                   patience=7, mode="min")]
        num_epochs = -1
        extra = {"enable_progress_bar": True, "callbacks": callbacks}

    trainer = pl.Trainer(
        default_root_dir="/data.nst/smohr/text_anomalies",
        max_epochs=num_epochs,
        accelerator="gpu",
        devices=1,
        **extra,
    )
    if not tunning:
        #lr_finder = trainer.tuner.lr_find(model, data)
        #print(lr_finder.suggestion())
        #model.learning_rate = lr_finder.suggestion()
        pass
        
    # Fit model
    trainer.fit(
        model,
        data,
    )

    # Save model
    trainer.save_checkpoint(f"../data/doec/models/ssvae_4.ckpt")


if __name__ == "__main__":
    main()