"""Request/response DTOs for the `/v1/recommendations` endpoint.

Kept separate from `domain/entities` so the wire format (v1) can evolve
independently of the domain model.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MovieCandidate(BaseModel):
    """One candidate movie to score, as supplied by the caller."""

    movie_id: int
    title: str
    genres: list[str]


class RecommendationRequestBody(BaseModel):
    """Body of `POST /v1/recommendations`."""

    user_id: int
    candidates: list[MovieCandidate] = Field(min_length=1)
    top_k: int | None = None


class ScoredMovieResponse(BaseModel):
    """One ranked candidate in the response."""

    movie_id: int
    title: str
    genres: list[str]
    score: float


class RecommendationResponse(BaseModel):
    """Body of a successful `POST /v1/recommendations` response."""

    user_id: int
    items: list[ScoredMovieResponse]
