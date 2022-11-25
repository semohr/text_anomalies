import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class OldEnglishDataset(Dataset):
    """Dataset wrapper for the Dictionary of Old English Corpus"""

    def __init__(self, path, transform=None):
        """
        Parameters
        ----------
        path : str
            Path to the data file (parquet expected)
        """
        # Read Parquet file
        self.data = pd.read_parquet(
            path,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Indexing and return numpy array
        # Check index shape
        if isinstance(index, int):
            return self.data.iloc[index]["text"]
        return self.data.iloc[index]["text"].values
