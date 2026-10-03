# triton-inference
Inference module for my Nvidia Triton Server project

Start the application locally:

```sh
make run
```

The API is available at `http://127.0.0.1:8000`, with interactive documentation
at `http://127.0.0.1:8000/docs`.

Runtime settings are composed with Hydra from
`src/triton_inference/configs/inference/config.yaml`. Configs are included in
the Python package, so loading them does not depend on the working directory.

- `dependencies/dependencies.yaml`: Triton URL and vocabulary path.
- `adapters/recommender.yaml`: model name, input tensor names and datatypes,
  and output tensor name.
- `config.yaml`: FastAPI application settings and Hydra defaults.

`TRITON_URL`, `TRITON_MODEL_NAME` and `VOCABULARY_PATH` remain supported through
OmegaConf environment interpolation in YAML. Hydra overrides take precedence:

```python
from triton_inference.config import load_config
from triton_inference.main import create_app

config = load_config([
    "dependencies.triton_url=triton:8001",
    "dependencies.vocabulary_path=/data/vocabulary.json",
    "adapters.model_name=recommender_custom",
])
app = create_app(config)
```

The default ASGI application is `triton_inference.main:app`. Each application
loads its vocabulary and creates its Triton client on the first recommendation,
then reuses them for subsequent requests.
