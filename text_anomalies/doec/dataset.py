import pandas as pd
import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset


class DOECDataset(Dataset):
    """Dataset wrapper for the Dictionary of Old English Corpus"""

    def __init__(self, data: pd.DataFrame, tokenizer: Tokenizer):
        """
        Parameters
        ----------
        data : pd.DataFrame
            Data to use for the dataset
        tokenizer : DOECTokenizer
            Tokenizer to use for the dataset
        """
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # One hot from title_id
        class_idx = int(row["title_id"])

        class_idx = torch.tensor(class_idx, dtype=torch.long)

        return {
            "x": row["text"],
            "x_true": row["text"],
            "y_true": class_idx,
        }


    def collate_fn(self, data):
        x_batch = [row["x"] for row in data]
        x_true_batch = [row["x_true"] for row in data]
        y_true_batch = [row["y_true"] for row in data]
        
        encoding_x = self.tokenizer(
            x_batch,
            padding="longest",
            add_special_tokens=True,
            return_tensors="pt",
            return_attention_mask=False,
        )
        encoding_x_true = self.tokenizer(
            x_true_batch,
            padding="longest",
            add_special_tokens=True,
            return_tensors="pt",
            return_attention_mask=False,
        )

        return {
            "x": encoding_x["input_ids"].squeeze(0),
            "x_true": encoding_x_true["input_ids"].squeeze(0),
            "y_true": torch.stack(y_true_batch),
        }

