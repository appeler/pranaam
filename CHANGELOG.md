# Changelog

## [Unreleased]

## 0.8.0 - 2026-08-17

* Replace `pred_rel` with `estimate_muslim_name_pattern` and rename the model
  cache refresh option to the truthful `refresh_pinned`/`--refresh-pinned`.
* Abstain explicitly when normalized UTF-8 input exceeds the model byte limit.
* Return typed reference-population, label-source, calibration-population, and
  model provenance fields with every estimate.
* Write model metadata schema 2 while retaining a strict reader adapter for the
  immutable v3 schema 1 artifacts.
* Define pinned refreshes for local mirrors as reread-and-verify operations;
  local mirror files are never downloaded or replaced.
* Keep the training-only PyArrow dependency compatible with the current
  Streamlit interface dependency.
* Use the standard bounded uv build backend and explicit package version.

## 0.7.0 - 2026-08-16

* Introduce model v3, replacing the v1/v2 whole-word averaging architecture
  with ordered byte-level PyTorch convolutional models for Latin and
  Devanagari names.
* Return calibrated 0-to-1 name-pattern scores, explicit abstention and script
  support status, and immutable model provenance.
* Add grouped, non-overlapping training, validation, calibration, and evaluation
  splits with reproducible training reports.
* Document the external SEPRI audit and prohibit individual or consequential
  uses in the supported product scope.

## 0.6.0 - 2026-08-16

* Correct probability reporting by exposing the model's softmax output directly.
* Verify downloaded model files and store them in the user cache.
* Replace the TensorFlow runtime with parity-tested PyTorch safetensors hosted
  at a pinned `gojiberries/pranaam` Hugging Face revision.
* Cache English and Hindi models independently for thread-safe language
  switching.
* Batch predictions to keep inference memory bounded on large datasets.
* Make `latest=True` refresh an already loaded language model.
* Repair the command-line entry point and test the Streamlit interface.
* Adopt the py-canon package, CI, documentation, and release structure.
