"""Tests for `POST /v1/recommendations` (TDD scaffold, v1).

Pin down the contract before `app/api/v1/recommendations.py` is
implemented (see its TODOs):
    - 200 + `RecommendationResponse` on success, items ranked best score
      first, and the domain `RecommendationRequest` the handler builds is
      the one actually sent to the `RecommenderPort`.
    - 422 on a malformed body (enforced by FastAPI/pydantic already) or a
      domain-invariant violation (duplicate candidate `movie_id`,
      non-positive `top_k`, ...).
    - 404 when the `RecommenderPort` raises `UnknownTokenError` (user or
      candidate unknown to the vocabulary).

These tests only exercise the HTTP layer: the real `RecommenderPort` is
swapped for `FakeRecommender` via dependency override, so none of them
need a running Triton server. Until the handler is implemented, every
test that reaches it fails on the `NotImplementedError` stub raises -
that's the expected TDD red state.
"""

from __future__ import annotations

import pytest

from triton_inference.domain.entities import (
    Movie,
    Recommendation,
    RecommendationRequest,
)
from triton_inference.domain.exceptions import UnknownTokenError

ENDPOINT = "/v1/recommendations"


class FakeRecommender:
    """Test double for `RecommenderPort`: records the request it received
    and returns a canned `Recommendation`, or raises a canned error."""

    def __init__(
        self, result: Recommendation | None = None, error: Exception | None = None
    ):
        self.result = result
        self.error = error
        self.received: RecommendationRequest | None = None

    def recommend(self, request: RecommendationRequest) -> Recommendation:
        self.received = request
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def movie_payload(movie: Movie) -> dict:
    """JSON body for one `MovieCandidate`, built from a domain `Movie`."""
    return {
        "movie_id": movie.movie_id,
        "title": movie.title,
        "genres": list(movie.genres),
    }


class TestHappyPath:
    def test_returns_200_with_items_ranked_best_first(
        self, client, use_recommender, user, drama, action_comedy
    ):
        fake = FakeRecommender(
            result=Recommendation.from_scores(
                user, movies=[drama, action_comedy], scores=[0.1, 0.9]
            )
        )
        use_recommender(fake)

        response = client.post(
            ENDPOINT,
            json={
                "user_id": user.user_id,
                "candidates": [movie_payload(drama), movie_payload(action_comedy)],
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["user_id"] == user.user_id
        assert [item["movie_id"] for item in body["items"]] == [100, 200]
        assert body["items"][0]["score"] == pytest.approx(0.9)

    def test_forwards_a_matching_domain_request_to_the_recommender(
        self, client, use_recommender, user, drama, action_comedy
    ):
        fake = FakeRecommender(
            result=Recommendation.from_scores(user, movies=[drama], scores=[1.0])
        )
        use_recommender(fake)

        client.post(
            ENDPOINT,
            json={
                "user_id": user.user_id,
                "candidates": [movie_payload(drama), movie_payload(action_comedy)],
                "top_k": 1,
            },
        )

        assert fake.received is not None
        assert fake.received.user == user
        assert fake.received.candidates == (drama, action_comedy)
        assert fake.received.top_k == 1


class TestRequestValidation:
    def test_missing_user_id_returns_422(self, client, drama):
        response = client.post(ENDPOINT, json={"candidates": [movie_payload(drama)]})
        assert response.status_code == 422

    def test_empty_candidates_returns_422(self, client, user):
        response = client.post(
            ENDPOINT, json={"user_id": user.user_id, "candidates": []}
        )
        assert response.status_code == 422

    def test_duplicate_candidate_movie_ids_returns_422(
        self, client, use_recommender, user, drama
    ):
        use_recommender(FakeRecommender())

        response = client.post(
            ENDPOINT,
            json={
                "user_id": user.user_id,
                "candidates": [movie_payload(drama), movie_payload(drama)],
            },
        )

        assert response.status_code == 422

    @pytest.mark.parametrize("bad_top_k", [0, -1])
    def test_non_positive_top_k_returns_422(
        self, client, use_recommender, user, drama, bad_top_k
    ):
        use_recommender(FakeRecommender())

        response = client.post(
            ENDPOINT,
            json={
                "user_id": user.user_id,
                "candidates": [movie_payload(drama)],
                "top_k": bad_top_k,
            },
        )

        assert response.status_code == 422


class TestUnknownToken:
    def test_unknown_token_error_returns_404(
        self, client, use_recommender, user, drama
    ):
        use_recommender(FakeRecommender(error=UnknownTokenError()))

        response = client.post(
            ENDPOINT,
            json={"user_id": user.user_id, "candidates": [movie_payload(drama)]},
        )

        assert response.status_code == 404
