"""Tensor shapes, batching and score order at the Triton boundary."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from triton_inference.adapters.outbound.recommender import TritonRecommender
from triton_inference.domain.entities import InferenceBatch, Vocabulary


@pytest.mark.parametrize("batch_size", [1, 31, 32, 33, 64, 65])
@pytest.mark.parametrize("datatype", ["INT32", "INT64"])
def test_infer_batches_tensors_and_returns_scores_in_candidate_order(
    batch_size, datatype
):
    batch = InferenceBatch(
        user_ids=[2] * batch_size,
        movie_ids=list(reversed(range(batch_size))),
        genres=[[i, i + 1, i + 2] for i in range(batch_size)],
    )
    expected_scores = [float((i * 7) % 13) - 6 for i in range(batch_size)]
    chunk_sizes = [32] * (batch_size // 32)
    if batch_size % 32:
        chunk_sizes.append(batch_size % 32)
    responses = []
    offset = 0
    for size in chunk_sizes:
        response = MagicMock()
        response.as_numpy.return_value = np.array(
            expected_scores[offset : offset + size], dtype=np.float32
        ).reshape(size, 1)
        responses.append(response)
        offset += size
    client = MagicMock()
    client.infer.side_effect = responses
    recommender = TritonRecommender(
        client=client,
        vocabulary=Vocabulary(
            user_vocab={}, item_vocab={}, genre_vocab={}, max_genres=3
        ),
        model_name="custom_model",
        inputs={
            "user_ids": {"name": "users", "datatype": datatype},
            "movie_ids": {"name": "movies", "datatype": datatype},
            "genres": {"name": "categories", "datatype": datatype},
        },
        output_name="scores",
    )

    scores = recommender._infer(batch)

    assert scores == expected_scores
    assert all(isinstance(score, float) for score in scores)
    assert client.infer.call_count == len(chunk_sizes)
    offset = 0
    for call, size, response in zip(
        client.infer.call_args_list, chunk_sizes, responses
    ):
        arguments = call.kwargs
        assert arguments["model_name"] == "custom_model"
        assert [tensor.name() for tensor in arguments["inputs"]] == [
            "users",
            "movies",
            "categories",
        ]
        assert [list(tensor.shape()) for tensor in arguments["inputs"]] == [
            [size, 1],
            [size, 1],
            [size, 3],
        ]
        for field, tensor in zip(
            ("user_ids", "movie_ids", "genres"), arguments["inputs"]
        ):
            assert tensor.datatype() == datatype
            data = np.frombuffer(tensor._get_content(), dtype=datatype.lower()).reshape(
                tensor.shape()
            )
            expected = np.array(getattr(batch, field)[offset : offset + size]).reshape(
                tensor.shape()
            )
            np.testing.assert_array_equal(data, expected)
        assert len(arguments["outputs"]) == 1
        assert arguments["outputs"][0].name() == "scores"
        response.as_numpy.assert_called_once_with("scores")
        offset += size
