"""Port the API layer depends on to score recommendations.

The concrete implementation (Triton client + `Vocabulary` encoding, see
`domain/entities/encoding.py`) belongs in `adapters/` and is injected
through `app/dependencies.py`. Keeping this as a `Protocol` lets API tests
substitute a fake and stay independent of Triton.
"""

from typing import Protocol

from triton_inference.domain.entities import Recommendation, RecommendationRequest


class RecommenderPort(Protocol):
    """Scores `request.candidates` for `request.user` and ranks them."""

    def recommend(self, request: RecommendationRequest) -> Recommendation:
        #   encode `request` via `Vocabulary.encode_*` /
        #   `InferenceBatch.from_request`, call the Triton server (gRPC or
        #   HTTP), and decode the response into a `Recommendation` via
        #   `Recommendation.from_scores`.
        ...
