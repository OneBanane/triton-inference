"""FastAPI application entrypoint (TDD scaffold).

TODO: add a startup hook that loads the `Vocabulary` (see
  `domain/entities/encoding.py`) and builds the Triton-backed
  `RecommenderPort` (see `app/ports.py`) once `adapters/` implements it,
  then override `get_recommender` (e.g. via `app.dependency_overrides` or
  by stashing the instance on `app.state`) so production requests use it
  instead of raising `NotImplementedError`.
"""

from fastapi import FastAPI

from triton_inference.app.api.v1.recommendations import (
    router as recommendations_router,
)


def create_app() -> FastAPI:
    app = FastAPI(title="triton-inference")
    app.include_router(recommendations_router)
    return app


app = create_app()
