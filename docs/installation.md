# Installation

## Requirements

* Python 3.10 or 3.11 (TensorFlow 2.14.1 compatibility requirement)
* TensorFlow 2.14.1 (automatically installed)

:::{note}
Python 3.12+ is not currently supported due to TensorFlow availability constraints.
:::

## Standard Installation

We strongly recommend installing pranaam inside a Python virtual environment. (see [venv documentation](https://docs.python.org/3/library/venv.html#creating-virtual-environments))

Install pranaam using pip:

```bash
pip install pranaam
```

This installs TensorFlow 2.14.1, which is known to work correctly with the models.

## Installation Options

For development work:

```bash
pip install -e .[dev]
```

For testing:

```bash
pip install -e .[test]
```

For documentation building:

```bash
pip install -e .[docs]
```

For all optional dependencies:

```bash
pip install -e .[all]
```

## TensorFlow Compatibility

The package requires TensorFlow 2.14.1 with Keras 2.14.0 for model compatibility. If you encounter compatibility issues:

```bash
pip install 'pranaam[tensorflow-compat]'
```

## Model Downloads

Models are downloaded on first use and stored in the operating system's user cache
directory. Downloads are installed only after both model files match their pinned
SHA-256 checksums. Concurrent processes share a cache lock, and a failed refresh
does not replace a verified cached model.

The default Harvard Dataverse endpoint may challenge automated clients. To use a
trusted mirror that serves the same model archive, set:

```bash
export PRANAAM_MODEL_URL="https://example.org/eng_and_hindi_models_v2.tar.gz"
```

Ensure you have:

* Stable internet connection
* At least 500MB free disk space
* Access to the configured model host

## Verification

Test your installation:

```python
import pranaam
result = pranaam.pred_rel("Shah Rukh Khan")
print(result)
```

If successful, you should see a pandas DataFrame with prediction results.

## Troubleshooting

### Common Issues

**TensorFlow/Keras Compatibility Errors**

Error: `"Keras 3 only supports V3 .keras files and legacy H5 format files"`

Solution: Install with `pip install 'pranaam[tensorflow-compat]'`

**Model Download Issues**

Error: Network timeouts or download failures

Solution: Check the connection to the configured model host. If Harvard Dataverse
returns an automated-client challenge, set `PRANAAM_MODEL_URL` to a trusted mirror.
The mirror must contain the original model files because checksum mismatches are
rejected.

**Import Errors**

Error: `pkg_resources` deprecation warnings

Solution: Already fixed in v0.1.0 (uses `importlib.resources`)

For additional help, please check our [GitHub Issues](https://github.com/appeler/pranaam/issues).
