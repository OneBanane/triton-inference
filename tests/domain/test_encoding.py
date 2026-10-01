"""Tests for Vocabulary and InferenceBatch (TDD scaffold).

They pin down the tensor contract of the served model: one row per
(user, movie) pair, genres right-padded with `num_genres`.
"""

import pytest
from pydantic import ValidationError

from triton_inference.domain.entities import (
    InferenceBatch,
    Movie,
    RecommendationRequest,
    User,
    Vocabulary,
)
from triton_inference.domain.exceptions import UnknownTokenError


class TestVocabularyValidation:
    def test_rejects_non_contiguous_indices(self):
        with pytest.raises(ValidationError):
            Vocabulary(
                user_vocab={10: 0, 20: 2},  # gap at 1
                item_vocab={100: 0},
                genre_vocab={"Drama": 0},
                max_genres=1,
            )

    def test_rejects_duplicate_indices(self):
        with pytest.raises(ValidationError):
            Vocabulary(
                user_vocab={10: 0},
                item_vocab={100: 0, 200: 0},
                genre_vocab={"Drama": 0},
                max_genres=1,
            )

    def test_rejects_indices_not_starting_at_zero(self):
        with pytest.raises(ValidationError):
            Vocabulary(
                user_vocab={10: 0},
                item_vocab={100: 0},
                genre_vocab={"Drama": 1},
                max_genres=1,
            )

    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_non_positive_max_genres(self, bad):
        with pytest.raises(ValidationError):
            Vocabulary(
                user_vocab={10: 0},
                item_vocab={100: 0},
                genre_vocab={"Drama": 0},
                max_genres=bad,
            )

    def test_is_immutable(self, vocabulary):
        with pytest.raises(ValidationError):
            vocabulary.max_genres = 5


class TestVocabularySizes:
    def test_sizes(self, vocabulary):
        assert vocabulary.num_users == 3
        assert vocabulary.num_items == 3
        assert vocabulary.num_genres == 4

    def test_genre_pad_index_is_one_past_last_genre(self, vocabulary):
        assert vocabulary.genre_pad_index == 4


class TestVocabularyEncoding:
    def test_encode_user(self, vocabulary):
        assert vocabulary.encode_user(User(user_id=20)) == 1

    def test_encode_unknown_user_raises(self, vocabulary):
        with pytest.raises(UnknownTokenError):
            vocabulary.encode_user(User(user_id=999))

    def test_unknown_token_error_is_a_key_error(self, vocabulary):
        with pytest.raises(KeyError):
            vocabulary.encode_user(User(user_id=999))

    def test_encode_movie(self, vocabulary, action_thriller):
        assert vocabulary.encode_movie(action_thriller) == 2

    def test_encode_unknown_movie_raises(self, vocabulary):
        stranger = Movie(movie_id=999, title="Unknown", genres=("Drama",))
        with pytest.raises(UnknownTokenError):
            vocabulary.encode_movie(stranger)

    def test_encode_genres_full_row(self, vocabulary, action_thriller):
        assert vocabulary.encode_genres(action_thriller) == [0, 3]

    def test_encode_genres_pads_short_row(self, vocabulary, drama):
        assert vocabulary.encode_genres(drama) == [2, 4]

    def test_encode_unknown_genre_raises(self, vocabulary):
        movie = Movie(movie_id=100, title="T", genres=("Western",))
        with pytest.raises(UnknownTokenError):
            vocabulary.encode_genres(movie)

    def test_encode_too_many_genres_raises(self, vocabulary):
        movie = Movie(movie_id=100, title="T", genres=("Action", "Comedy", "Drama"))
        with pytest.raises(ValueError):
            vocabulary.encode_genres(movie)


class TestInferenceBatchValidation:
    def test_accepts_aligned_columns(self):
        batch = InferenceBatch(
            user_ids=[0, 0], movie_ids=[1, 2], genres=[[0, 4], [1, 2]]
        )
        assert len(batch) == 2

    def test_rejects_empty_batch(self):
        with pytest.raises(ValidationError):
            InferenceBatch(user_ids=[], movie_ids=[], genres=[])

    def test_rejects_column_length_mismatch(self):
        with pytest.raises(ValidationError):
            InferenceBatch(user_ids=[0, 0], movie_ids=[1], genres=[[0], [1]])

    def test_rejects_ragged_genres(self):
        with pytest.raises(ValidationError):
            InferenceBatch(user_ids=[0, 0], movie_ids=[1, 2], genres=[[0, 4], [1]])

    def test_rejects_negative_indices(self):
        with pytest.raises(ValidationError):
            InferenceBatch(user_ids=[-1], movie_ids=[1], genres=[[0]])

    def test_is_immutable(self):
        batch = InferenceBatch(user_ids=[0], movie_ids=[1], genres=[[0]])
        with pytest.raises(ValidationError):
            batch.user_ids = [5]


class TestInferenceBatchFromRequest:
    def test_one_row_per_candidate_in_request_order(
        self, vocabulary, drama, action_comedy
    ):
        request = RecommendationRequest(
            user=User(user_id=20), candidates=[drama, action_comedy]
        )
        batch = InferenceBatch.from_request(request, vocabulary)
        assert batch.user_ids == [1, 1]
        assert batch.movie_ids == [1, 0]
        assert batch.genres == [[2, 4], [0, 1]]

    def test_ignores_top_k(self, vocabulary, user, drama, action_comedy):
        request = RecommendationRequest(
            user=user, candidates=[drama, action_comedy], top_k=1
        )
        assert len(InferenceBatch.from_request(request, vocabulary)) == 2

    def test_unknown_user_propagates(self, vocabulary, drama):
        request = RecommendationRequest(user=User(user_id=999), candidates=[drama])
        with pytest.raises(UnknownTokenError):
            InferenceBatch.from_request(request, vocabulary)

    def test_unknown_candidate_propagates(self, vocabulary, user):
        stranger = Movie(movie_id=999, title="Unknown", genres=("Drama",))
        request = RecommendationRequest(user=user, candidates=[stranger])
        with pytest.raises(UnknownTokenError):
            InferenceBatch.from_request(request, vocabulary)
