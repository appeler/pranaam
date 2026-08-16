# Changelog

## Unreleased

## 0.6.0 - 2026-08-16

* Correct probability reporting by using the model's calibrated output directly.
* Verify downloaded model files and store them in the user cache.
* Replace the TensorFlow runtime with parity-tested PyTorch safetensors hosted
  at a pinned `gojiberries/pranaam` Hugging Face revision.
* Cache English and Hindi models independently for thread-safe language
  switching.
* Batch predictions to keep inference memory bounded on large datasets.
* Make `latest=True` refresh an already loaded language model.
* Repair the command-line entry point and test the Streamlit interface.
* Adopt the py-canon package, CI, documentation, and release structure.
