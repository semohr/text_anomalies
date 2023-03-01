from typing import Iterator

from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    decoders,
    PreTrainedTokenizerFast,
)


def create_and_train_tokenizer(iterator: Iterator[str]) -> Tokenizer:
    """
    Create a tokenizer from an iterator of text. The tokenizer is trained on the text using The WordPieceTraining algorithm. We also add special tokens to the tokenizer, namely [UNK], [PAD], [CLS], [SEP], [MASK].
    """

    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.Lowercase(), normalizers.NFKD()]
    )

    # 2. Pre-tokenization
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # 3. Model
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
        pad_token="[PAD]",
        unk_token="[UNK]",
    )

    return tokenizer
