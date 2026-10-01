"""Triton-backed `RecommenderPort` adapter (gRPC).

Wires `domain/entities/encoding.py` (`Vocabulary`, `InferenceBatch`) to the
`recommender_onnx` model served by Triton Inference Server over gRPC. See
`triton/models/recommender_onnx/config.pbtxt` for the model's tensor
contract: INT64 `user_id` [N], `movie_id` [N], `genres` [N, max_genres],
FP32 `output` [N] (`max_batch_size: 0`, so no implicit batch dimension).
"""

from __future__ import annotations

import numpy as np
import tritonclient.grpc as grpcclient

from triton_inference.domain.entities import (
    InferenceBatch,
    Recommendation,
    RecommendationRequest,
    Vocabulary,
)

DEFAULT_MODEL_NAME = "recommender_onnx"


class TritonRecommender:
    """`RecommenderPort` implementation backed by a Triton gRPC client.

    Encodes `request` via `vocabulary` into an `InferenceBatch`, sends one
    gRPC `infer` call to the model, and decodes the response into a ranked
    `Recommendation`. Propagates `UnknownTokenError` / `ValueError` raised
    while encoding `request`.
    """

    def __init__(
        self,
        client: grpcclient.InferenceServerClient,
        vocabulary: Vocabulary,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        self._client = client
        self._vocabulary = vocabulary
        self._model_name = model_name

    def recommend(self, request: RecommendationRequest) -> Recommendation:
        batch = InferenceBatch.from_request(request, self._vocabulary)
        scores = self._infer(batch)
        return Recommendation.from_scores(
            user=request.user,
            movies=request.candidates,
            scores=scores,
            top_k=request.top_k,
        )

    def _infer(self, batch: InferenceBatch) -> list[float]:
        user_id = np.asarray(batch.user_ids, dtype=np.int64)
        movie_id = np.asarray(batch.movie_ids, dtype=np.int64)
        genres = np.asarray(batch.genres, dtype=np.int64)

        inputs = [
            grpcclient.InferInput("user_id", user_id.shape, "INT64"),
            grpcclient.InferInput("movie_id", movie_id.shape, "INT64"),
            grpcclient.InferInput("genres", genres.shape, "INT64"),
        ]
        inputs[0].set_data_from_numpy(user_id)
        inputs[1].set_data_from_numpy(movie_id)
        inputs[2].set_data_from_numpy(genres)

        output = grpcclient.InferRequestedOutput("output")

        response = self._client.infer(
            model_name=self._model_name, inputs=inputs, outputs=[output]
        )

        return response.as_numpy("output").tolist()
