"""Shared fixtures for API tests."""

import pytest
from fastapi.testclient import TestClient

from triton_inference.app.dependencies import get_recommender
from triton_inference.main import app
from triton_inference.domain.entities import Movie, User


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def use_recommender():
    """Register `fake` as the `get_recommender` dependency for this test."""

    def _use(fake) -> None:
        app.dependency_overrides[get_recommender] = lambda: fake

    yield _use
    app.dependency_overrides.pop(get_recommender, None)


@pytest.fixture
def user() -> User:
    return User(user_id=10)


@pytest.fixture
def drama() -> Movie:
    return Movie(movie_id=200, title="Drama (1999)", genres=("Drama",))


@pytest.fixture
def action_comedy() -> Movie:
    return Movie(
        movie_id=100, title="Action Comedy (2000)", genres=("Action", "Comedy")
    )
