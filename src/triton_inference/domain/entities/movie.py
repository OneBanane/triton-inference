"""Movie entity: a recommendable item (one row of MovieLens `movies.csv`)."""

from __future__ import annotations

from pydantic import BaseModel, field_validator, ConfigDict


class Movie(BaseModel):
    """A MovieLens movie.

    Expected contract (TDD scaffold, see tests/domain/test_movie.py):
        - `movie_id` is the raw MovieLens id and must be a positive integer
        - `title` is stripped of surrounding whitespace and must not be blank
        - `genres` is a non-empty tuple of unique, non-blank genre tokens
          that do not contain the MovieLens separator "|"
        - instances are immutable and hashable
    """

    model_config = ConfigDict(frozen=True)

    movie_id: int
    title: str
    genres: tuple[str, ...]

    @field_validator("movie_id")
    @classmethod
    def _check_movie_id(cls, movie_id: int) -> int:
        if movie_id <= 0:
            raise ValueError("movie_id must be a positive integer")

        return movie_id

    @field_validator("title")
    @classmethod
    def _check_title(cls, title: str) -> str:
        title = title.strip()
        if len(title) == 0:
            raise ValueError("title must be not a blank")

        return title

    @field_validator("genres")
    @classmethod
    def _check_genres(cls, genres: tuple[str, ...]) -> tuple[str, ...]:
        if len(genres) == 0:
            raise ValueError("genres must be a non-empty")

        if len(genres) != len(set(genres)):
            raise ValueError("genres must contain unique tokens")

        for genre in genres:
            genre = genre.strip()
            if genre == "":
                raise ValueError("genres must contain non-blank tokens")

            if "|" in genre:
                raise ValueError("genres must contain the MovieLens separator '|'")

        return genres
