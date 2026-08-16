# Changelog

## Unreleased

* Correct probability reporting by using the model's calibrated output directly.
* Verify downloaded model files and store them in the user cache.
* Replace the TensorFlow runtime with parity-tested PyTorch safetensors hosted
  at a pinned `gojiberries/pranaam` Hugging Face revision.
* Cache English and Hindi models independently for thread-safe language
  switching.
* Make `latest=True` refresh an already loaded language model.
* Repair the command-line entry point and test the Streamlit interface.
* Adopt the py-canon package, CI, documentation, and release structure.
