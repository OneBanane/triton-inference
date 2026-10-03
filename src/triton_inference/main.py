"""FastAPI application entrypoint configured with Hydra."""

from copy import deepcopy

from fastapi import FastAPI
from omegaconf import DictConfig, OmegaConf

from triton_inference.app.api.v1.recommendations import (
    router as recommendations_router,
)
from triton_inference.app.dependencies import LazyRecommender
from triton_inference.config import load_config


def create_app(config: DictConfig | None = None) -> FastAPI:
    config = load_config() if config is None else deepcopy(config)
    OmegaConf.resolve(config)
    app = FastAPI(title=config.app.title)
    app.state.config = config
    app.state.recommender = LazyRecommender(config)
    app.include_router(recommendations_router)
    return app


app = create_app()
