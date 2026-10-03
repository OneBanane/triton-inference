"""Triton-backed `RecommenderPort` adapter (gRPC).

Wires `domain/entities/encoding.py` (`Vocabulary`, `InferenceBatch`) to the
`recommender_onnx` model served by Triton Inference Server over gRPC. See
`triton/models/recommender_onnx/config.pbtxt` for the model's tensor
contract: INT64 `user_id` [B, 1], `movie_id` [B, 1],
`genres` [B, max_genres], FP32 `output` [B, 1], with at most 32 rows per call.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import triton_to_np_dtype

from triton_inference.domain.entities import (
    InferenceBatch,
    Recommendation,
    RecommendationRequest,
    Vocabulary,
)

MAX_BATCH_SIZE = 32


class TritonRecommender:
    """`RecommenderPort` implementation backed by a Triton gRPC client.

    Encodes `request` via `vocabulary` into an `InferenceBatch`, sends
    gRPC `infer` calls of at most 32 rows, and combines scores into a ranked
    `Recommendation`. Propagates `UnknownTokenError` / `ValueError` raised
    while encoding `request`.
    """

    def __init__(
        self,
        client: grpcclient.InferenceServerClient,
        vocabulary: Vocabulary,
        model_name: str,
        inputs: Mapping[str, Mapping[str, str]],
        output_name: str,
    ) -> None:
        self._client = client
        self._vocabulary = vocabulary
        self._model_name = model_name
        self._inputs = inputs
        self._output_name = output_name

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
        """Return one score per candidate, preserving the batch's row order."""
        scores = []
        output = grpcclient.InferRequestedOutput(self._output_name)
        for start in range(0, len(batch), MAX_BATCH_SIZE):
            inputs = []
            for field, tensor in self._inputs.items():
                data = np.asarray(
                    getattr(batch, field)[start : start + MAX_BATCH_SIZE],
                    dtype=triton_to_np_dtype(tensor["datatype"]),
                )
                if field in ("user_ids", "movie_ids"):
                    data = data.reshape(-1, 1)
                infer_input = grpcclient.InferInput(
                    tensor["name"], data.shape, tensor["datatype"]
                )
                infer_input.set_data_from_numpy(data)
                inputs.append(infer_input)

            response = self._client.infer(
                model_name=self._model_name, inputs=inputs, outputs=[output]
            )
            scores.extend(response.as_numpy(self._output_name).reshape(-1).tolist())

        return scores
