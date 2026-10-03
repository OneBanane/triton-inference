"""Entities that bridge domain objects and the Triton model's tensor contract.

The served ONNX model (see `triton_models/models/recommender_onnx/config.pbtxt`)
takes three INT64 inputs per call, one row per (user, movie) pair:
    user_id  [B, 1]          contiguous user index
    movie_id [B, 1]          contiguous item index
    genres   [B, max_genres] genre indices, right-padded with `num_genres`
and returns FP32 `output` [B, 1] with one score per pair. The adapter shapes
the ID columns and splits the encoded candidates into batches of at most 32.
"""

from __future__ import annotations

from itertools import chain

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    field_validator,
    model_validator,
)

from triton_inference.domain.exceptions import UnknownTokenError

from .movie import Movie
from .recommendation import RecommendationRequest
from .user import User


class Vocabulary(BaseModel):
    """Raw-id -> contiguous-index maps the model was trained with.

    Mirrors `MovieLensDataset.{user_vocab,item_vocab,genre_vocab,max_genres}`.

    Expected contract (TDD scaffold, see tests/domain/test_encoding.py):
        - each vocab maps its keys onto exactly `0..len(vocab)-1`
          (contiguous, no duplicates)
        - `max_genres` is a positive integer
        - instances are immutable
    """

    model_config = ConfigDict(frozen=True)

    user_vocab: dict[int, int]
    item_vocab: dict[int, int]
    genre_vocab: dict[str, int]
    max_genres: int

    @field_validator("user_vocab", "item_vocab", "genre_vocab")
    @classmethod
    def _check_vocab(cls, vocab: dict, info: ValidationInfo) -> dict:
        if sorted(vocab.values()) != list(range(len(vocab))):
            raise ValueError(
                f"{info.field_name} must map its keys onto exactly `0..len(vocab)-1`"
            )

        return vocab

    @field_validator("max_genres")
    @classmethod
    def _check_max_genres(cls, max_genres: int) -> int:
        if max_genres <= 0:
            raise ValueError("max_genres must be a positive integer")

        return max_genres

    @property
    def num_users(self) -> int:
        return len(self.user_vocab)

    @property
    def num_items(self) -> int:
        return len(self.item_vocab)

    @property
    def num_genres(self) -> int:
        return len(self.genre_vocab)

    @property
    def genre_pad_index(self) -> int:
        """Index used to right-pad genres: one past the last real genre."""
        return self.num_genres

    def encode_user(self, user: User) -> int:
        """Map the raw user id to its index; `UnknownTokenError` if unknown."""
        index = self.user_vocab.get(user.user_id, -1)
        if index == -1:
            raise UnknownTokenError("unknown user")

        return index

    def encode_movie(self, movie: Movie) -> int:
        """Map the raw movie id to its index; `UnknownTokenError` if unknown."""
        index = self.item_vocab.get(movie.movie_id, -1)
        if index == -1:
            raise UnknownTokenError("unknown movie")

        return index

    def encode_genres(self, movie: Movie) -> list[int]:
        """Genre indices of `movie`, in order, padded to `max_genres`.

        Raises `UnknownTokenError` for an unknown genre and `ValueError` if
        the movie has more than `max_genres` genres.
        """
        if len(movie.genres) > self.max_genres:
            raise ValueError("movie has more than `max_genres` genres")

        res = []
        for genre in movie.genres:
            index = self.genre_vocab.get(genre, -1)
            if index == -1:
                raise UnknownTokenError("unknown genre")
            res.append(index)

        return res + (self.max_genres - len(res)) * [self.genre_pad_index]


class InferenceBatch(BaseModel):
    """Model-ready input: parallel columns, one row per (user, movie) pair.

    Expected contract:
        - at least one row
        - `user_ids`, `movie_ids` and `genres` have the same length
        - every `genres` row has the same length (rectangular) and all
          indices are non-negative
        - instances are immutable
    """

    model_config = ConfigDict(frozen=True)

    user_ids: list[int]
    movie_ids: list[int]
    genres: list[list[int]]

    @model_validator(mode="after")
    def _check_batch(self) -> InferenceBatch:
        if not self.user_ids:
            raise ValueError("batch must contain at least one row")

        if not (len(self.user_ids) == len(self.movie_ids) == len(self.genres)):
            raise ValueError(
                "`user_ids`, `movie_ids` and `genres` must have the same length"
            )

        if len({len(row) for row in self.genres}) > 1:
            raise ValueError("every `genres` row must have the same length")

        indices = chain(self.user_ids, self.movie_ids, *self.genres)
        if any(index < 0 for index in indices):
            raise ValueError("all indices must be non-negative")

        return self

    def __len__(self) -> int:
        return len(self.user_ids)

    @classmethod
    def from_request(
        cls, request: RecommendationRequest, vocabulary: Vocabulary
    ) -> InferenceBatch:
        """Encode every candidate of `request` paired with the request's user.

        Row order follows `request.candidates`. Propagates
        `UnknownTokenError` / `ValueError` from `Vocabulary`.
        """
        user_ids = []
        movie_ids = []
        genres = []

        for candidate in request.candidates:
            user_ids.append(vocabulary.encode_user(request.user))
            movie_ids.append(vocabulary.encode_movie(candidate))
            genres.append(vocabulary.encode_genres(candidate))

        return cls(user_ids=user_ids, movie_ids=movie_ids, genres=genres)
