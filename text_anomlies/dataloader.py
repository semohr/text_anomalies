import os
from typing import Tuple, Union
from torchtext.data.datasets_utils import (
    _wrap_split_argument,
    _create_dataset_directory,
)
from torchdata.datapipes.iter import FileOpener, IterableWrapper


@_create_dataset_directory(dataset_name="OLD_ENGLISH")
@_wrap_split_argument(("train", "test"))
def OLD_ENGLISH(
    root: str = "data/doec/html",
    split: str = Union[Tuple[str], str],
) -> IterableWrapper:
    """Dictionary of Old English Corpus (DOEC) dataset

    See  http://hdl.handle.net/20.500.12024/2488
    for more details. The dataset has

    Args:
        split:  split: split or splits to be returned. Can be a string or tuple of strings. Default: (`train`, `test`)
    """

    # Load all files but "changes.html file
    data = FileOpener(os.path.join(root, "*.html"), mode="rb", length=-1)

    # Filter out "changes.html" file
    data = data.filter(lambda x: "changes.html" not in x[0])
