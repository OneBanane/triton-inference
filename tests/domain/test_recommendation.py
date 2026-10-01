"""Tests for the recommendation entities (TDD scaffold):
ScoredMovie, RecommendationRequest and Recommendation."""

import math

import pytest
from pydantic import ValidationError

from triton_inference.domain.entities import (
    Recommendation,
    RecommendationRequest,
    ScoredMovie,
    User,
)


class TestScoredMovie:
    def test_stores_fields(self, drama):
        scored = ScoredMovie(movie=drama, score=1.5)
        assert scored.movie == drama
        assert scored.score == 1.5

    def test_allows_negative_scores(self, drama):
        assert ScoredMovie(movie=drama, score=-3.0).score == -3.0

    @pytest.mark.parametrize("bad_score", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_score(self, drama, bad_score):
        with pytest.raises(ValidationError):
            ScoredMovie(movie=drama, score=bad_score)

    def test_is_immutable(self, drama):
        scored = ScoredMovie(movie=drama, score=1.0)
        with pytest.raises(ValidationError):
            scored.score = 2.0


class TestRecommendationRequest:
    def test_top_k_defaults_to_none(self, user, drama):
        assert RecommendationRequest(user=user, candidates=[drama]).top_k is None

    def test_candidates_become_tuple(self, user, drama, action_comedy):
        request = RecommendationRequest(user=user, candidates=[drama, action_comedy])
        assert request.candidates == (drama, action_comedy)

    def test_rejects_empty_candidates(self, user):
        with pytest.raises(ValidationError):
            RecommendationRequest(user=user, candidates=[])

    def test_rejects_duplicate_candidates(self, user, drama):
        with pytest.raises(ValidationError):
            RecommendationRequest(user=user, candidates=[drama, drama])

    @pytest.mark.parametrize("bad_top_k", [0, -1])
    def test_rejects_non_positive_top_k(self, user, drama, bad_top_k):
        with pytest.raises(ValidationError):
            RecommendationRequest(user=user, candidates=[drama], top_k=bad_top_k)

    def test_allows_top_k_larger_than_candidates(self, user, drama):
        request = RecommendationRequest(user=user, candidates=[drama], top_k=50)
        assert request.top_k == 50

    def test_is_immutable(self, user, drama):
        request = RecommendationRequest(user=user, candidates=[drama])
        with pytest.raises(ValidationError):
            request.top_k = 3


class TestRecommendation:
    def test_accepts_items_sorted_best_first(self, user, drama, action_comedy):
        rec = Recommendation(
            user=user,
            items=[
                ScoredMovie(movie=drama, score=2.0),
                ScoredMovie(movie=action_comedy, score=1.0),
            ],
        )
        assert len(rec.items) == 2

    def test_accepts_tied_scores(self, user, drama, action_comedy):
        Recommendation(
            user=user,
            items=[
                ScoredMovie(movie=drama, score=1.0),
                ScoredMovie(movie=action_comedy, score=1.0),
            ],
        )

    def test_rejects_items_not_sorted(self, user, drama, action_comedy):
        with pytest.raises(ValidationError):
            Recommendation(
                user=user,
                items=[
                    ScoredMovie(movie=action_comedy, score=1.0),
                    ScoredMovie(movie=drama, score=2.0),
                ],
            )

    def test_rejects_duplicate_movies(self, user, drama):
        with pytest.raises(ValidationError):
            Recommendation(
                user=user,
                items=[
                    ScoredMovie(movie=drama, score=2.0),
                    ScoredMovie(movie=drama, score=1.0),
                ],
            )

    def test_allows_empty_items(self, user):
        assert Recommendation(user=user, items=[]).items == ()

    def test_is_immutable(self, user):
        rec = Recommendation(user=user, items=[])
        with pytest.raises(ValidationError):
            rec.user = User(user_id=99)

    def test_movie_ids_follow_ranking(self, user, drama, action_comedy):
        rec = Recommendation(
            user=user,
            items=[
                ScoredMovie(movie=drama, score=2.0),
                ScoredMovie(movie=action_comedy, score=1.0),
            ],
        )
        assert rec.movie_ids == (200, 100)


class TestFromScores:
    def test_ranks_best_first(self, user, drama, action_comedy, action_thriller):
        rec = Recommendation.from_scores(
            user,
            movies=[drama, action_comedy, action_thriller],
            scores=[0.1, 0.9, 0.5],
        )
        assert rec.movie_ids == (100, 300, 200)
        assert [item.score for item in rec.items] == [0.9, 0.5, 0.1]

    def test_keeps_only_top_k(self, user, drama, action_comedy, action_thriller):
        rec = Recommendation.from_scores(
            user,
            movies=[drama, action_comedy, action_thriller],
            scores=[0.1, 0.9, 0.5],
            top_k=2,
        )
        assert rec.movie_ids == (100, 300)

    def test_top_k_larger_than_candidates_returns_all(self, user, drama, action_comedy):
        rec = Recommendation.from_scores(
            user, movies=[drama, action_comedy], scores=[1.0, 2.0], top_k=10
        )
        assert len(rec.items) == 2

    def test_ties_keep_candidate_order(self, user, drama, action_comedy):
        rec = Recommendation.from_scores(
            user, movies=[drama, action_comedy], scores=[1.0, 1.0]
        )
        assert rec.movie_ids == (200, 100)

    def test_rejects_length_mismatch(self, user, drama, action_comedy):
        with pytest.raises(ValueError):
            Recommendation.from_scores(
                user, movies=[drama, action_comedy], scores=[1.0]
            )

    @pytest.mark.parametrize("bad_top_k", [0, -1])
    def test_rejects_non_positive_top_k(self, user, drama, bad_top_k):
        with pytest.raises(ValueError):
            Recommendation.from_scores(
                user, movies=[drama], scores=[1.0], top_k=bad_top_k
            )
