import pandas as pd
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast


class DOECDataset(Dataset):
    """Dataset wrapper for the Dictionary of Old English Corpus"""

<<<<<<< HEAD
    def __init__(self, data: pd.DataFrame, tokenizer: PreTrainedTokenizerFast):
=======
    def __init__(self, data: pd.DataFrame, tokenizer: Tokenizer):
>>>>>>> e85f155e6d95cb3758f129ce8fa3799fe1d52601
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
        encoding = self.tokenizer(
            row.text,
            max_length=512,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt",
            return_attention_mask=False,
        )
        labels = "TODO"
        return {
            "input_ids": encoding["input_ids"].flatten(),
<<<<<<< HEAD
            "labels": labels,
=======
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": "TODO",
>>>>>>> e85f155e6d95cb3758f129ce8fa3799fe1d52601
        }
