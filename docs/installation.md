# Installation

Pranaam supports Python 3.11 and newer. Install the published package in a
virtual environment:

```bash
python -m pip install pranaam
```

PyTorch, safetensors, Hugging Face Hub support, and the other runtime
dependencies are installed with the package. The first estimate downloads
only the requested language's `safetensors` weights and inference metadata from
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
eng/metadata.json
hin/model.safetensors
hin/metadata.json
```

Local mirror files must match the same release checksums. A missing, corrupt,
or unavailable artifact causes estimation to fail without replacing an already
loaded language model. `refresh_pinned=True` and `--refresh-pinned` reread and
verify these local files; they never download into or replace the mirror.

## Metadata schemas

New training runs write metadata schema 2. It stores reference population,
label source, calibration population, training seed, and normalization as
typed provenance fields. The shipped immutable v3 artifacts use schema 1.
Pranaam accepts schema 1 only through an internal adapter for that published v3
shape, then exposes the same typed provenance as schema 2. Other schema 1
documents fail validation.
