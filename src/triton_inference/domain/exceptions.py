"""Domain-level errors."""


class UnknownTokenError(KeyError):
    """A raw id / genre token is missing from the vocabulary.

    Subclasses `KeyError` to mirror `triton.training.dataset.MovieLensDataset`,
    which also refuses to silently map cold-start ids.
    """
