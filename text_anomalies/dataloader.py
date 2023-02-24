import random
import numpy as np
import pandas as pd
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


class SyntheicDataset(Dataset):
    """Dataset wrapper for synthetic data"""

    def __init__(
        self,
        generate,
        fractions,
        total=100_000,
        transform=None,
    ):
        """
        Parameters
        ----------
        generate : array of function
            Function to generate a normal text sentence.
        fractions : array of float
            Fractions of anomalies in the dataset.
        total : int, optional
            Total number of sentences in the dataset. Defaults to 100_000.
        transform : function, optional
        """
        # Set seed
        random.seed(42)

        # Normalize fractions
        fractions = np.array(fractions) / np.sum(fractions)

        # Number of anomalies
        nSentences = np.array(fractions * total, dtype=int)
        # Round to total (a bit hacky but works)
        nSentences[-1] = total - np.sum(nSentences[:-1])

        # Generate data
        texts = np.concatenate(
            [
                [generate[i]() for _ in range(nSentences[i])]
                for i in range(len(fractions))
            ]
        )
        labels = np.concatenate(
            [[i for _ in range(nSentences[i])] for i in range(len(fractions))]
        )

        self.data = pd.DataFrame(
            {
                "text": texts,
                "label": labels,
            }
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Indexing and return numpy array
        # Check index shape
        if isinstance(index, int):
            return self.data.iloc[index]["text"]
        return self.data.iloc[index]["text"].values
