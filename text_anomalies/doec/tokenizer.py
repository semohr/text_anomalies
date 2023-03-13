from typing import Iterator

from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    decoders,
)
from transformers import PreTrainedTokenizerFast


def create_and_train_tokenizer(iterator: Iterator[str]) -> PreTrainedTokenizerFast:
    """
    Create a tokenizer from an iterator of text. The tokenizer is trained on the text using The WordPieceTraining algorithm. We also add special tokens to the tokenizer, namely [UNK], [PAD], [CLS], [SEP], [MASK].
    """

    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.Lowercase()]
    )

    # Pre-tokenization
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Model
    # e.g., the tokenization or merging of characters or sub-words into larger logical components.
    trainer = trainers.WordPieceTrainer(
        vocab_size=30_000,
        special_tokens=["[UNK]", "[PAD]"],
        min_frequency=2,
        continuing_subword_prefix="##",
    )

    tokenizer.train_from_iterator(iterator, trainer=trainer)

    tokenizer.decoder = decoders.WordPiece(prefix="##")

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
    )

    return tokenizer
