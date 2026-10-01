"""`POST /v1/recommendations` (TDD scaffold).

Contract pinned down by `tests/app/api/v1/test_recommendations.py`:
    - 200 + `RecommendationResponse`: candidates scored by the injected
      `RecommenderPort`, ranked best score first, truncated to `top_k`
      when given.
    - 422: the request body fails schema validation, or building the
      domain `RecommendationRequest` fails its invariants (duplicate
      candidate `movie_id`, non-positive `top_k`, ...).
    - 404: the user or a candidate (movie id / genre) is unknown to the
      vocabulary (`UnknownTokenError` raised by the `RecommenderPort`).
"""

from anyio import to_thread
from fastapi import APIRouter, Depends, HTTPException

from triton_inference.app.dependencies import get_recommender
from triton_inference.app.ports import RecommenderPort
from triton_inference.domain.entities import User, Movie, RecommendationRequest
from triton_inference.domain.exceptions import UnknownTokenError

from .schemas import (
    RecommendationRequestBody,
    RecommendationResponse,
    ScoredMovieResponse,
)

router = APIRouter()


@router.post("/v1/recommendations", response_model=RecommendationResponse)
async def recommend(
    body: RecommendationRequestBody,
    recommender: RecommenderPort = Depends(get_recommender),  # noqa: B008
) -> RecommendationResponse:
    try:
        user: User = User(user_id=body.user_id)
        candidates: list[Movie] = [
            Movie(movie_id=c.movie_id, title=c.title, genres=c.genres)
            for c in body.candidates
        ]
        request: RecommendationRequest = RecommendationRequest(
            user=user, candidates=candidates, top_k=body.top_k
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    try:
        recommendation = await to_thread.run_sync(recommender.recommend, request)
    except UnknownTokenError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return RecommendationResponse(
        user_id=recommendation.user.user_id,
        items=[
            ScoredMovieResponse(
                movie_id=item.movie.movie_id,
                title=item.movie.title,
                genres=list(item.movie.genres),
                score=item.score,
            )
            for item in recommendation.items
        ],
    )
