# Changelog

## [Unreleased]

## 0.9.0 - 2026-08-19

Breaking release: the result shape and call signature change. There are no
backward-compatibility aliases.

* Adopt appeler inference contract 1.1, score form
  (https://github.com/appeler/appellation): results carry
  `inference_contract_version`, `estimate_type`, `result_form`, `target`,
  `input_scope`, boolean `scored` and `abstained`, `model_id`,
  `calibration_status`, `calibration_reference`, `uncertainty_method`, and
  `uncertainty_level` alongside the calibrated score.
* Remove the `name_pattern_estimate` label. For a binary target the score
  already carries the whole distribution, and the cutoff that turns it into a
  decision depends on the caller's costs. `uncertain-score` abstention goes
  with it: a mid-range score is a real, calibrated answer and is now returned.
* Take a DataFrame and a name column, with every option keyword-only. A
  single string, list, or Series still works and returns the same columns.
* Abstain instead of raising on blank, missing, and non-text names, which are
  data rather than programming errors, using `missing-name` and `no-letters`.
* Add Monte Carlo dropout intervals through `uncertainty_level` and
  `mc_iterations`, reported as `muslim_score_mc_mean`, `_mc_std`,
  `_mc_lower`, and `_mc_upper`.
* Add `prior` to reweight scores from the calibration base rate, now reported
  as `reference_prior`, to a target population's base rate.
* Rename `calibration_population` to the contract's `calibration_reference`
  and drop the redundant `input_truncated` column, whose abstention reason
  already carries the fact.
* Delete the unreachable v1 and v2 model module, its tests, and the adhoc
  conversion and comparison scripts that depended on it.
* State in the documentation that the negative class pools every non-Muslim
  naming pattern, and that the Bihar sources contain too few Christian, Sikh,
  Buddhist, and Jain names to separate them.
* Keep abstaining rows missing under `prior` and Monte Carlo summaries. Prior
  shifting turned a missing score into a certain 1.0, because a missing value
  failed the finite-odds test and fell through to the saturated branch.

Released without the independent second-model review the project's release
process normally requires: both reviewers were unavailable at release time.
The maintainer accepted that gap knowingly. Every other gate ran, and the
prior-shift defect above was found by a self-review afterwards.

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
* Exclude generated documentation output from source distributions.
* Apply the serving byte limit to every training split and paired-audit
  reconstruction before fitting, calibration, or evaluation, and record
  excluded row counts.

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
