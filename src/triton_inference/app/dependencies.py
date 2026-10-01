"""FastAPI dependency providers for the API layer.

Builds the real `RecommenderPort` lazily and caches it for the lifetime of
the process: a `Vocabulary` loaded from `VOCABULARY_PATH` and a
`TritonRecommender` (see `adapters/outbound/recommender.py`) backed by a
gRPC client pointed at `TRITON_URL`. Tests override `get_recommender`
(`app.dependency_overrides[get_recommender] = ...`) with a fake
`RecommenderPort` instead of exercising this wiring.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import tritonclient.grpc as grpcclient

from triton_inference.adapters.outbound.recommender import (
    DEFAULT_MODEL_NAME,
    TritonRecommender,
)
from triton_inference.app.ports import RecommenderPort
from triton_inference.domain.entities import (
    Recommendation,
    RecommendationRequest,
    Vocabulary,
)

TRITON_URL_ENV = "TRITON_URL"
TRITON_MODEL_NAME_ENV = "TRITON_MODEL_NAME"
VOCABULARY_PATH_ENV = "VOCABULARY_PATH"

DEFAULT_TRITON_URL = "localhost:8001"
DEFAULT_VOCABULARY_PATH = "vocabulary.json"


@lru_cache(maxsize=1)
def _load_vocabulary() -> Vocabulary:
    path = Path(os.environ.get(VOCABULARY_PATH_ENV, DEFAULT_VOCABULARY_PATH))
    return Vocabulary.model_validate_json(path.read_text())


@lru_cache(maxsize=1)
def _build_recommender() -> TritonRecommender:
    url = os.environ.get(TRITON_URL_ENV, DEFAULT_TRITON_URL)
    model_name = os.environ.get(TRITON_MODEL_NAME_ENV, DEFAULT_MODEL_NAME)
    client = grpcclient.InferenceServerClient(url=url)
    return TritonRecommender(
        client=client, vocabulary=_load_vocabulary(), model_name=model_name
    )


class _LazyRecommender:
    """Defers building the real `TritonRecommender` until first use.

    FastAPI calls `Depends(get_recommender)` while resolving every
    request's dependencies, even one whose body fails validation before
    the handler runs. Keeping this object cheap to construct means those
    requests never pay for (or fail on) loading the vocabulary / opening
    the gRPC channel; only a request that actually reaches `.recommend()`
    triggers (and then reuses, via `_build_recommender`'s cache) that I/O.
    """

    def recommend(self, request: RecommendationRequest) -> Recommendation:
        return _build_recommender().recommend(request)


_lazy_recommender = _LazyRecommender()


def get_recommender() -> RecommenderPort:
    return _lazy_recommender
