"""Triton-backed `RecommenderPort` adapter (gRPC).

Wires `domain/entities/encoding.py` (`Vocabulary`, `InferenceBatch`) to the
`recommender_onnx` model served by Triton Inference Server over gRPC. See
`triton/models/recommender_onnx/config.pbtxt` for the model's tensor
contract: INT64 `user_id` [N], `movie_id` [N], `genres` [N, max_genres],
FP32 `output` [N] (`max_batch_size: 0`, so no implicit batch dimension).
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
        inputs = []
        for field, tensor in self._inputs.items():
            data = np.asarray(
                getattr(batch, field), dtype=triton_to_np_dtype(tensor["datatype"])
            )
            infer_input = grpcclient.InferInput(
                tensor["name"], data.shape, tensor["datatype"]
            )
            infer_input.set_data_from_numpy(data)
            inputs.append(infer_input)

        output = grpcclient.InferRequestedOutput(self._output_name)

        response = self._client.infer(
            model_name=self._model_name, inputs=inputs, outputs=[output]
        )

        return response.as_numpy(self._output_name).tolist()
