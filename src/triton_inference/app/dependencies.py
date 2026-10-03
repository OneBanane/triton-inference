"""FastAPI dependency providers for the API layer.

Builds the real `RecommenderPort` lazily and caches it per application,
using the vocabulary path, Triton URL and adapter settings from Hydra.
Tests override `get_recommender`
(`app.dependency_overrides[get_recommender] = ...`) with a fake
`RecommenderPort` instead of exercising this wiring.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import tritonclient.grpc as grpcclient
from fastapi import Request
from omegaconf import DictConfig, OmegaConf

from triton_inference.adapters.outbound.recommender import TritonRecommender
from triton_inference.app.ports import RecommenderPort
from triton_inference.domain.entities import (
    Recommendation,
    RecommendationRequest,
    Vocabulary,
)


class LazyRecommender:
    """Defers building the real `TritonRecommender` until first use.

    FastAPI calls `Depends(get_recommender)` while resolving every
    request's dependencies, even one whose body fails validation before
    the handler runs. Keeping this object cheap to construct means those
    requests never pay for (or fail on) loading the vocabulary / opening
    the gRPC channel; only a request that actually reaches `.recommend()`
    triggers (and then reuses) that I/O.
    """

    def __init__(self, config: DictConfig) -> None:
        self._config = config

    @cached_property
    def _recommender(self) -> TritonRecommender:
        path = Path(self._config.dependencies.vocabulary_path)
        vocabulary = Vocabulary.model_validate_json(path.read_text())
        client = grpcclient.InferenceServerClient(
            url=self._config.dependencies.triton_url
        )
        return TritonRecommender(
            client=client,
            vocabulary=vocabulary,
            model_name=self._config.adapters.model_name,
            inputs=OmegaConf.to_container(self._config.adapters.inputs, resolve=True),
            output_name=self._config.adapters.output_name,
        )

    def recommend(self, request: RecommendationRequest) -> Recommendation:
        return self._recommender.recommend(request)


def get_recommender(request: Request) -> RecommenderPort:
    return request.app.state.recommender
