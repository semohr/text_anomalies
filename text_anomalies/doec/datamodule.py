import pathlib
import os
from typing import Optional
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from transformers import PreTrainedTokenizerFast

from .tokenizer import create_and_train_tokenizer
from .preprocess import preprocess_data
from .dataset import DOECDataset
from .preprocess import preprocess_data


class DOECDataModule(LightningDataModule):
    """
    Data module for the DOEC dataset
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
    ):
        """
        Parameters
        ----------
        data_dir : str
            Path to the data directory containing the raw or processed DOEC dataset.
            If put in the default location, this should be `../../data/doec`.
        """
        super().__init__()
        self.data_dir = pathlib.Path(data_dir)
        self.data_dir_raw = pathlib.Path(data_dir) / "raw"
        self.batch_size = batch_size

    def prepare_data(self):
        """
        Prepare the data for training.
        """
        # Check if "sgml-corpus" folder exists and
        # if any *.sgml files are present
        if (
            not os.path.exists(self.data_dir_raw / "sgml-corpus")
            or len(list((self.data_dir_raw / "sgml-corpus").glob("*.sgml"))) == 0
        ):
            raise FileNotFoundError(
                "The DOEC dataset is not present in the data directory. Please download it (http://hdl.handle.net/20.500.12024/2488) and place it in the data directory."
            )

        # Check if "doec.parquet" file exists
        if not os.path.exists(self.data_dir / "doec.parquet"):
            # Create parquet file
            data = preprocess_data(self.data_dir_raw / "sgml-corpus")
            data.to_parquet(self.data_dir / "doec.parquet")

        # Check if tokenizer is present
        if not os.path.exists(self.data_dir / "tokenizer") or not os.path.exists(
            self.data_dir / "tokenizer" / "tokenizer.json"
        ):
            # Create tokenizer
            data = pd.read_parquet(self.data_dir / "doec.parquet")
            tokenizer = create_and_train_tokenizer(iter(data.text))
            tokenizer.save_pretrained(self.data_dir / "tokenizer")

    def setup(self, stage: Optional[str] = None):
        """
        Split the data into train, validation and test sets.
        """
        # Load data, tokenizer and create dataset
        self.data = pd.read_parquet(self.data_dir / "doec.parquet")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            self.data_dir / "tokenizer"
        )

        self.dataset = DOECDataset(
            self.data,
            self.tokenizer,
        )
        # Split dataset
        split = [0.8, 0.18, 0.02]
        self.train_ds, self.val_ds, self.test_ds = random_split(self.dataset, split)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=True,
            collate_fn=self.dataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=False,
            collate_fn=self.dataset.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=False,
            collate_fn=self.dataset.collate_fn,
        )

    @property
    def num_classes(self):
        return self.dataset.data["title_id"].nunique()