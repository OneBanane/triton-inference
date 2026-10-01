"""Tests for triton_inference.domain.entities.Movie (TDD scaffold)."""

import pytest
from pydantic import ValidationError

from triton_inference.domain.entities import Movie


def test_stores_fields():
    movie = Movie(movie_id=1, title="Toy Story (1995)", genres=["Adventure", "Comedy"])
    assert movie.movie_id == 1
    assert movie.title == "Toy Story (1995)"
    assert movie.genres == ("Adventure", "Comedy")


@pytest.mark.parametrize("bad_id", [0, -5])
def test_rejects_non_positive_id(bad_id):
    with pytest.raises(ValidationError):
        Movie(movie_id=bad_id, title="T", genres=("Drama",))


def test_strips_title_whitespace():
    assert Movie(movie_id=1, title="  Heat (1995) \n", genres=("Crime",)).title == (
        "Heat (1995)"
    )


@pytest.mark.parametrize("bad_title", ["", "   "])
def test_rejects_blank_title(bad_title):
    with pytest.raises(ValidationError):
        Movie(movie_id=1, title=bad_title, genres=("Drama",))


def test_rejects_empty_genres():
    with pytest.raises(ValidationError):
        Movie(movie_id=1, title="T", genres=())


@pytest.mark.parametrize(
    "bad_genres",
    [
        ("Drama", "Drama"),  # duplicate
        ("Drama", ""),  # blank token
        ("Drama", "  "),  # whitespace-only token
        ("Action|Comedy",),  # separator leaked into a token
    ],
)
def test_rejects_invalid_genre_tokens(bad_genres):
    with pytest.raises(ValidationError):
        Movie(movie_id=1, title="T", genres=bad_genres)


def test_is_immutable():
    movie = Movie(movie_id=1, title="T", genres=("Drama",))
    with pytest.raises(ValidationError):
        movie.title = "Other"


def test_equal_by_value_and_hashable():
    a = Movie(movie_id=1, title="T", genres=("Drama",))
    b = Movie(movie_id=1, title="T", genres=("Drama",))
    assert a == b
    assert len({a, b}) == 1
