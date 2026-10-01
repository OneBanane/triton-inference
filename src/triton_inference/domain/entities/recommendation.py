"""Request / response entities of the recommendation use case."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, field_validator, ConfigDict

from .movie import Movie
from .user import User


class ScoredMovie(BaseModel):
    """A movie together with the model's match score for some user.

    Expected contract (TDD scaffold, see tests/domain/test_recommendation.py):
        - `score` must be a finite float (no NaN / +-inf); it is an
          unbounded dot-product style score, not a probability
        - instances are immutable
    """

    model_config = ConfigDict(frozen=True)

    movie: Movie
    score: float

    @field_validator("score")
    @classmethod
    def _check_score(cls, score: float) -> float:
        if not math.isfinite(score):
            raise ValueError("score should be a finite number")

        return score


class RecommendationRequest(BaseModel):
    """Ask the model to score `candidates` for `user`.

    Expected contract:
        - `candidates` is a non-empty tuple of movies with unique `movie_id`s
        - `top_k` is None (return every candidate) or a positive integer;
          a `top_k` larger than the number of candidates is allowed
        - instances are immutable
    """

    model_config = ConfigDict(frozen=True)

    user: User
    candidates: tuple[Movie, ...]
    top_k: int | None = None

    @field_validator("candidates")
    @classmethod
    def _check_candidates(cls, candidates: tuple[Movie, ...]) -> tuple[Movie, ...]:
        if candidates is None or len(candidates) == 0:
            raise ValueError("candidates should be a non-empty")

        movie_ids = [m.movie_id for m in candidates]
        if len(movie_ids) != len(set(movie_ids)):
            raise ValueError("candidates should contain only unique move ids")

        return candidates

    @field_validator("top_k")
    @classmethod
    def _check_top_k(cls, top_k: int) -> int:
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k should be greater than 0")

        return top_k


class Recommendation(BaseModel):
    """Ranked answer for one user.

    Expected contract:
        - `items` are sorted by `score`, best first (ties are allowed)
        - the same movie may not appear twice
        - instances are immutable
    """

    model_config = ConfigDict(frozen=True)

    user: User
    items: tuple[ScoredMovie, ...]

    @field_validator("items")
    @classmethod
    def _check_items(cls, items: tuple[ScoredMovie, ...]) -> tuple[ScoredMovie, ...]:
        ids = [i.movie.movie_id for i in items]
        if len(ids) != len(set(ids)):
            raise ValueError("the same movie may not appear twice")

        scores = [i.score for i in items]
        if any(a < b for a, b in zip(scores, scores[1:])):
            raise ValueError("items must be sorted by score, best first")

        return items

    @property
    def movie_ids(self) -> tuple[int, ...]:
        """Raw movie ids in ranking order."""
        return tuple([i.movie.movie_id for i in self.items])

    @classmethod
    def from_scores(
        cls,
        user: User,
        movies: Sequence[Movie],
        scores: Sequence[float],
        top_k: int | None = None,
    ) -> Recommendation:
        """Pair `movies[i]` with `scores[i]`, rank best-first, keep `top_k`.

        Ties keep the original candidate order (stable sort). Raises
        `ValueError` if `movies` and `scores` differ in length.
        """
        if len(movies) != len(scores):
            raise ValueError("movies and score differ in length")

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k should be greater than 0")

        scored_movies: Sequence[ScoredMovie] = []

        for movie, score in zip(movies, scores):
            scored_movies.append(ScoredMovie(movie=movie, score=score))

        ranked_movies: Sequence[ScoredMovie] = sorted(
            scored_movies, key=lambda x: x.score, reverse=True
        )

        if top_k is not None:
            ranked_movies = ranked_movies[:top_k]

        return cls(user=user, items=ranked_movies, top_k=top_k)
