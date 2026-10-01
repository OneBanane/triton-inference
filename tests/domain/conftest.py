"""Shared fixtures for domain entity tests."""

import pytest

from triton_inference.domain.entities import Movie, User, Vocabulary


@pytest.fixture
def user() -> User:
    return User(user_id=10)


@pytest.fixture
def action_comedy() -> Movie:
    return Movie(
        movie_id=100, title="Action Comedy (2000)", genres=("Action", "Comedy")
    )


@pytest.fixture
def drama() -> Movie:
    return Movie(movie_id=200, title="Drama (1999)", genres=("Drama",))


@pytest.fixture
def action_thriller() -> Movie:
    return Movie(
        movie_id=300, title="Action Thriller (2010)", genres=("Action", "Thriller")
    )


@pytest.fixture
def vocabulary() -> Vocabulary:
    """Raw ids are non-contiguous on purpose, like MovieLens."""
    return Vocabulary(
        user_vocab={10: 0, 20: 1, 30: 2},
        item_vocab={100: 0, 200: 1, 300: 2},
        genre_vocab={"Action": 0, "Comedy": 1, "Drama": 2, "Thriller": 3},
        max_genres=2,
    )
