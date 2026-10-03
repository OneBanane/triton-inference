"""Compose the application's Hydra configuration without changing the cwd."""

from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


def load_config(overrides: Sequence[str] = ()) -> DictConfig:
    """Load packaged defaults and apply Hydra overrides and environment values."""
    config_dir = Path(__file__).resolve().parent / "configs" / "inference"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="config", overrides=list(overrides))
    OmegaConf.resolve(config)
    return config
