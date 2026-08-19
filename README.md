# Pranaam

[![CI](https://github.com/appeler/pranaam/actions/workflows/ci.yml/badge.svg)](https://github.com/appeler/pranaam/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pranaam.svg)](https://pypi.org/project/pranaam/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://appeler.github.io/pranaam/)
[![Downloads](https://static.pepy.tech/badge/pranaam)](https://pepy.tech/project/pranaam)

Pranaam returns a calibrated probability that an English- or Hindi-script
name follows patterns associated with Muslim names in its training data. It
does not observe or establish a person's religion.

Results follow the appeler [inference contract](https://github.com/appeler/appellation),
score form: one probability on a 0 to 1 scale, explicit abstention with a
machine-readable reason instead of a fabricated score, and provenance columns
identifying the exact artifacts used. Pranaam returns no label. For a binary
target the score already carries the whole distribution, and the cutoff that
would turn it into a decision depends on the costs of your analysis, not on
this package.

Pranaam is for validated aggregate research. **Do not use it to label
individuals, make consequential decisions, determine eligibility, target
people, or replace self-identified information.**

Model v3 uses compact byte-level PyTorch models. Unlike the v1 and v2
whole-word model, it retains local character order, represents every UTF-8
byte without an unknown-word token, does not average padded embeddings into
each representation, and learns spelling fragments that generalize to unseen names.

The historical v1 model was trained on complete recorded name strings. Model
v2 migrated those same weights to newer serialization and runtime formats; it
was not a new training run. Both versions nevertheless averaged whole-word
embeddings, so accepting a full name did not preserve word order.

## Install

```bash
pip install pranaam
```

Python 3.11 or newer is required. The first estimate downloads small,
checksum-verified `safetensors` artifacts from an immutable revision of
[`gojiberries/pranaam`](https://huggingface.co/gojiberries/pranaam).

## Use

```python
from pranaam import estimate_muslim_name_pattern

result = estimate_muslim_name_pattern(
    ["Shah Rukh Khan", "Amitabh Bachchan", "محمد خان"],
    lang="eng",
)
print(result)
```

`estimate_muslim_name_pattern` takes a DataFrame and the name of its name
column, and returns a copy with the estimate columns appended:

```python
import pandas as pd

frame = pd.DataFrame({"full_name": ["Shah Rukh Khan"], "row_id": [1]})
result = estimate_muslim_name_pattern(frame, "full_name")
```

A single name, a list, or a pandas Series also works, and options are
keyword-only. Use `lang="hin"` for Devanagari names.

The target-specific and provenance columns are:

| Column | Meaning |
|---|---|
| `name` | Original input |
| `muslim_score` | Platt-calibrated probability from 0 to 1; missing when Pranaam abstains |
| `scored` | Whether the model produced a usable score for this row |
| `abstained` | Whether Pranaam declined to score |
| `abstention_reason` | `missing-name`, `no-letters`, `unsupported-script`, `input-truncated`, or missing |
| `script_supported` | Whether every input letter is supported by the selected model |
| `normalized_utf8_bytes` | Byte length after the model's Unicode and whitespace normalization |
| `reference_prior` | Base rate the shipped calibration is anchored to |
| `target_prior` | Base rate requested through `prior`, or missing |
| `reference_population` | Population against which the selected model's score is calibrated |
| `label_source` | Observed variables used to construct the selected model's labels |
| `calibration_reference` | Held-out population used for Platt calibration |
| `model_language` | Selected language model |
| `model_metadata_schema` | Version of the metadata document loaded with the model |
| `model_version` | Model-family version |
| `model_revision` | Immutable Hugging Face commit used for inference |
| `model_max_name_bytes` | Maximum normalized UTF-8 content bytes accepted without truncation |

Rows also carry the contract's shared metadata: `inference_contract_version`,
`estimate_type`, `result_form`, `target`, `input_scope`, `model_id`,
`calibration_status`, `uncertainty_method`, and `uncertainty_level`.

The English model supports Latin letters and the Hindi model supports
Devanagari letters. Selecting the wrong model therefore produces an explicit
unsupported-script abstention rather than a fabricated score. A blank or
non-text cell abstains with `missing-name` rather than raising, because a
missing name in a column of data is data, not a programming error.

### Uncertainty

`uncertainty_level` reports a central interval from Monte Carlo dropout,
which describes how unstable the model's own score is, not sampling error in
the training data:

```python
result = estimate_muslim_name_pattern(
    ["Shah Rukh Khan"], uncertainty_level=0.9, mc_iterations=64
)
result[["muslim_score", "muslim_score_mc_lower", "muslim_score_mc_upper"]]
```

### Adapting to a different base rate

The shipped calibration is anchored to the base rate of its evaluation
split, reported in `reference_prior`. When your population's share of
Muslim-associated names differs, `prior` reweights the posterior odds
accordingly:

```python
result = estimate_muslim_name_pattern(["Shah Rukh Khan"], prior=0.30)
```

This assumes only the class balance differs between the two populations,
not the naming patterns within each class.

### The target is binary, and its negative class is everything else

The models separate Muslim-associated naming patterns from everything else.
They do not distinguish Hindu, Christian, Sikh, Buddhist, or Jain naming
patterns from one another: in the Bihar sources behind the labels, the 2011
census records 82.7 percent Hindu and 16.9 percent Muslim, with Christians
at 0.12 percent and Sikhs, Buddhists, and Jains at roughly 0.02 percent
each. There are too few names from those communities to learn a separate
class, and Indian Christian names overlap heavily with the majority. Read
`not-muslim-associated` as "Hindu or other", not as a clean religious
partition.

The byte limit applies after Unicode NFKC normalization, case folding, and
whitespace collapsing. Inputs longer than `model_max_name_bytes` are not
silently scored from a truncated prefix: they return `input_truncated=True`,
`abstention_reason="input-truncated"`, and a missing score.

New training runs write model metadata schema 2, which records the typed
population and label provenance directly. The immutable v3 release uses schema
1; Pranaam validates that exact legacy shape and adapts it to the same result
columns without changing scoring.

The command-line interface exposes the same result:

```bash
pranaam --input "Shah Rukh Khan" --lang eng
pranaam --input "Shah Rukh Khan" --uncertainty-level 0.9 --prior 0.3
```

## Evaluation

### Pranaam v0.6.0 audit

On all 92,897 directly labeled SEPRI household heads, v0.6.0 achieved:

- Accuracy: **96.43%**
- Muslim precision: **87.49%**
- Muslim recall: **73.74%**
- Muslim F1: **0.800**
- Recall on names not overlapping the translated land corpus: **69.10%**

The last measure uses exact normalized-name overlap. This external audit showed
why overall accuracy and the old random-row notebook results were insufficient:
the model missed more Muslim names when names were not represented in the land
corpus.

### Model v3

The released v2 and new v3 English pipelines were compared on the same
18,133-row SEPRI evaluation partition. Normalized names do not cross training,
validation, calibration, and evaluation partitions.

| Model | Accuracy | Muslim precision | Muslim recall | Muslim F1 | Brier | 10-bin ECE |
|---|---:|---:|---:|---:|---:|---:|
| v2 (Pranaam 0.6.0) | 96.51% | 87.75% | 74.14% | 0.804 | 0.0357 | 0.0395 |
| v3 | 97.46% | 90.29% | 82.49% | 0.862 | 0.0205 | 0.0052 |

With the default abstention rule, English v3 covers 96.54% of evaluation rows and is
98.54% accurate on retained estimates. Hindi v3 was evaluated on a disjoint
152,390-name grouped land-record test partition: Muslim precision 94.30%,
recall 93.05%, F1 0.937, and Brier score 0.0116. The Hindi result is an
in-source evaluation and should not be interpreted as national performance.

A paired audit also recalibrated v2 on v3's 13,665-row calibration partition.
Against that stronger baseline, v3 improved accuracy by 1.19 percentage points
(95% name-cluster bootstrap interval: 0.95 to 1.43), Muslim recall by 15.62
points (13.55 to 17.75), Muslim F1 by 0.086 (0.071 to 0.103), and Brier score by
0.0122 (0.0106 to 0.0139). Muslim precision was 2.04 points lower (-3.44 to
-0.67) because recalibrated v2 used a more conservative operating point. On
2,737 rows for which every word was outside v2's vocabulary, recall rose from
0% to 63.85%.

These results support v3 on the available SEPRI population, not universal
superiority. The v3 pipeline changes architecture, training data, and
calibration together, so this comparison does not identify the architecture's
effect alone. The evaluation partition was held out from parameter fitting and
calibration but was inspected during architecture development; it is therefore
developmental evidence rather than a pristine confirmatory test. See the
reproducible [paired audit](scripts/adhoc/compare_model_v2_v3.py) and its
[aggregate report](scripts/adhoc/v2_v3_comparison.json).

## Data and limitations

The models combine Bihar land-record names carrying caste/community-derived
silver labels with authorized SEPRI household-head data for the English model.
Conflicting labels for the same normalized land name are removed. SEPRI names
are assigned to deterministic, non-overlapping train, validation, calibration,
and test partitions.

Names are imperfect and culturally contingent proxies. Recorded caste,
household religion, transliteration, OCR, geography, gender, and time can all
introduce systematic error. Scores may be poorly calibrated outside the
evaluated populations. Validation against self-identified information at the
appropriate aggregate level remains the user's responsibility.

The training sources are Bihari, and the models are weakest on names from
Muslim communities elsewhere in India or the diaspora. Both `Salman Rushdie`
and `Azim Premji` score below 0.1 under the English model, because Kashmiri
and Gujarati Ismaili surnames barely appear in Bihar land records. Monte
Carlo dropout does not rescue these cases: it reports how unstable the model
is, not whether the name resembles anything the model was trained on, so an
out-of-region name can receive a low score and a narrow interval at the same
time. Treat regional coverage as a validation question for your own sample.

Raw personal names are not published with the package or model. Hugging Face
contains only weights, non-identifying training reports, metadata, and the
model card.

## Development

```bash
uv sync --all-groups
make ci
make docs
uv build
```

The reproducible v3 entry point is [`training/train_v3.py`](training/train_v3.py).
It reads authorized local source data and writes only weights and aggregate
reports.

## Authors

Rajashekar Chintalapati, Aaditya Dar, and Gaurav Sood.

## License

The package is released under the [MIT License](LICENSE). The responsible-use
requirements above describe the supported scope of the model.
