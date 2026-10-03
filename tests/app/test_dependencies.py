"""Application configuration reaches the lazy Triton adapter."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from triton_inference.app import dependencies
from triton_inference.config import load_config
from triton_inference.domain.entities import Vocabulary
from triton_inference.main import create_app


@pytest.fixture
def vocabulary_path(tmp_path):
    path = tmp_path / "vocabulary.json"
    vocabulary = Vocabulary(
        user_vocab={10: 0},
        item_vocab={200: 0, 100: 1},
        genre_vocab={"Drama": 0, "Action": 1, "Comedy": 2},
        max_genres=2,
    )
    path.write_text(vocabulary.model_dump_json())
    return path


@pytest.fixture
def triton_client_factory(monkeypatch):
    factory = MagicMock()
    factory.return_value.infer.return_value.as_numpy.return_value = np.array(
        [0.1, 0.9], dtype=np.float32
    )
    monkeypatch.setattr(dependencies.grpcclient, "InferenceServerClient", factory)
    return factory


def test_invalid_request_does_not_load_vocabulary_or_create_client(
    tmp_path, triton_client_factory
):
    config = load_config(
        [
            f"dependencies.vocabulary_path={tmp_path / 'missing.json'}",
        ]
    )
    app = create_app(config)

    with TestClient(app) as client:
        response = client.post(
            "/v1/recommendations", json={"user_id": 10, "candidates": []}
        )

    assert response.status_code == 422
    triton_client_factory.assert_not_called()


def test_requests_use_configured_triton_contract_and_reuse_client(
    vocabulary_path, triton_client_factory
):
    config = load_config(
        [
            f"dependencies.vocabulary_path={vocabulary_path}",
            "dependencies.triton_url=custom:9001",
            "adapters.model_name=custom_model",
            "adapters.inputs.user_ids.name=users",
            "adapters.inputs.movie_ids.name=movies",
            "adapters.inputs.genres.name=categories",
            "adapters.inputs.user_ids.datatype=INT32",
            "adapters.inputs.movie_ids.datatype=INT32",
            "adapters.inputs.genres.datatype=INT32",
            "adapters.output_name=scores",
            "app.title=custom-title",
        ]
    )
    app = create_app(config)
    payload = {
        "user_id": 10,
        "candidates": [
            {"movie_id": 200, "title": "Drama", "genres": ["Drama"]},
            {
                "movie_id": 100,
                "title": "Comedy",
                "genres": ["Action", "Comedy"],
            },
        ],
    }
    triton_client_factory.assert_not_called()

    with TestClient(app) as client:
        first_response = client.post("/v1/recommendations", json=payload)
        # A cached vocabulary must still work after its file disappears.
        vocabulary_path.rename(vocabulary_path.with_suffix(".backup"))
        second_response = client.post("/v1/recommendations", json=payload)

    assert app.title == "custom-title"
    assert first_response.status_code == second_response.status_code == 200
    assert [item["movie_id"] for item in first_response.json()["items"]] == [100, 200]
    triton_client_factory.assert_called_once_with(url="custom:9001")
    triton_client = triton_client_factory.return_value
    assert triton_client.infer.call_count == 2
    arguments = triton_client.infer.call_args.kwargs
    assert arguments["model_name"] == "custom_model"
    assert [tensor.name() for tensor in arguments["inputs"]] == [
        "users",
        "movies",
        "categories",
    ]
    assert [tensor.datatype() for tensor in arguments["inputs"]] == ["INT32"] * 3
    assert [list(tensor.shape()) for tensor in arguments["inputs"]] == [
        [2],
        [2],
        [2, 2],
    ]
    assert arguments["outputs"][0].name() == "scores"
    triton_client.infer.return_value.as_numpy.assert_called_with("scores")


def test_applications_have_independent_configs_and_clients(
    vocabulary_path, triton_client_factory
):
    config = load_config([f"dependencies.vocabulary_path={vocabulary_path}"])
    config.dependencies.triton_url = "first:8001"
    config.adapters.model_name = "first_model"
    first_app = create_app(config)
    config.dependencies.triton_url = "second:8001"
    config.adapters.model_name = "second_model"
    second_app = create_app(config)
    payload = {
        "user_id": 10,
        "candidates": [{"movie_id": 200, "title": "Drama", "genres": ["Drama"]}],
    }
    triton_client_factory.return_value.infer.return_value.as_numpy.return_value = (
        np.array([0.5], dtype=np.float32)
    )

    with TestClient(first_app) as client:
        assert client.post("/v1/recommendations", json=payload).status_code == 200
    assert triton_client_factory.call_args.kwargs["url"] == "first:8001"
    assert (
        triton_client_factory.return_value.infer.call_args.kwargs["model_name"]
        == "first_model"
    )

    with TestClient(second_app) as client:
        assert client.post("/v1/recommendations", json=payload).status_code == 200
    assert triton_client_factory.call_count == 2
    assert triton_client_factory.call_args.kwargs["url"] == "second:8001"
    assert (
        triton_client_factory.return_value.infer.call_args.kwargs["model_name"]
        == "second_model"
    )
