# Installation

Pranaam supports Python 3.11 and newer. Install the published package in a
virtual environment:

```bash
python -m pip install pranaam
```

PyTorch, safetensors, Hugging Face Hub support, and the other runtime
dependencies are installed with the package. The first prediction downloads
only the requested language's weights and vocabulary from
[`gojiberries/pranaam`](https://huggingface.co/gojiberries/pranaam) at an
immutable revision. Pranaam verifies every file against a pinned SHA-256 digest
before loading it from the Hugging Face cache.

To work on a checkout, install the locked development environment with uv:

```bash
uv sync --all-groups
uv run pytest
```

## Streamlit app

The repository includes a local Streamlit interface. Install its optional
dependency and run the launcher from the repository root:

```bash
uv sync --extra streamlit
uv run python streamlit/run_app.py
```

The launcher opens the app at `http://localhost:8501`.

## Model cache and offline use

Once a language has been downloaded, Hugging Face can reuse its local cache.
Set `HF_HUB_OFFLINE=1` to prevent network access when the pinned files are
already cached.

For an explicitly managed mirror, set `PRANAAM_MODEL_DIR` to a directory with
the published repository layout:

```text
eng/model.safetensors
eng/vocabulary.txt
hin/model.safetensors
hin/vocabulary.txt
```

Local mirror files must match the same release checksums. A missing, corrupt,
or unavailable artifact causes prediction to fail without replacing an already
loaded language model.
