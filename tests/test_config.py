"""Hydra defaults, overrides and configuration discovery."""

import pytest

from triton_inference.config import load_config


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in ("TRITON_URL", "TRITON_MODEL_NAME", "VOCABULARY_PATH"):
        monkeypatch.delenv(name, raising=False)


def test_defaults_are_composed_from_config_groups():
    config = load_config()

    assert config.dependencies.triton_url == "localhost:8001"
    assert config.dependencies.vocabulary_path == "vocabulary.json"
    assert config.adapters.model_name == "recommender_onnx"
    assert config.app.title == "triton-inference"


def test_environment_values_are_resolved_when_loading(monkeypatch):
    monkeypatch.setenv("TRITON_URL", "remote:9001")
    monkeypatch.setenv("TRITON_MODEL_NAME", "remote_model")
    monkeypatch.setenv("VOCABULARY_PATH", "/data/remote_vocabulary.json")

    config = load_config()
    monkeypatch.setenv("TRITON_URL", "changed:9001")

    assert config.dependencies.triton_url == "remote:9001"
    assert config.dependencies.vocabulary_path == "/data/remote_vocabulary.json"
    assert config.adapters.model_name == "remote_model"


def test_hydra_overrides_take_precedence_over_environment(monkeypatch):
    monkeypatch.setenv("TRITON_URL", "environment:9001")
    monkeypatch.setenv("TRITON_MODEL_NAME", "environment_model")
    monkeypatch.setenv("VOCABULARY_PATH", "/environment/vocabulary.json")

    config = load_config(
        [
            "dependencies.triton_url=override:9001",
            "dependencies.vocabulary_path=/override/vocabulary.json",
            "adapters.model_name=override_model",
            "app.title=custom-title",
        ]
    )

    assert config.dependencies.triton_url == "override:9001"
    assert config.dependencies.vocabulary_path == "/override/vocabulary.json"
    assert config.adapters.model_name == "override_model"
    assert config.app.title == "custom-title"
    assert load_config().adapters.model_name == "environment_model"


def test_config_discovery_is_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert load_config().adapters.model_name == "recommender_onnx"
    assert tmp_path == tmp_path.cwd()
