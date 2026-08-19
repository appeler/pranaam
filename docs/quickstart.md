# Quick start

`estimate_muslim_name_pattern` accepts one name, a list of names, or a pandas
Series. It returns one row per name.

```python
import pranaam

names = ["Shah Rukh Khan", "Amitabh Bachchan", "Abdul Kalam"]
result = pranaam.estimate_muslim_name_pattern(names, lang="eng")
print(result)
```

The output columns are:

* `name`: the input name
* `muslim_score`: calibrated probability from 0 to 1, missing when Pranaam
  abstains. Pranaam returns no label: for a binary target the score carries
  the whole distribution, and the cutoff belongs to your analysis.
* `scored`, `abstained`, and `abstention_reason`: whether and why no score
  was returned
* `script_supported`: whether the selected model supports every input letter
* `normalized_utf8_bytes`: byte length after normalization
* `reference_prior` and `target_prior`: the base rate the calibration is
  anchored to, and the one requested through `prior`
* `reference_population`, `label_source`, and `calibration_reference`: the
  population and labeling scope of the score
* `model_language`, `model_metadata_schema`, `model_version`, `model_revision`,
  and `model_max_name_bytes`: model provenance and support boundary
* the contract's shared metadata, including `inference_contract_version`,
  `result_form`, `target`, `calibration_status`, and `uncertainty_method`

Pass `lang="hin"` for names written in Hindi:

```python
hindi_names = ["शाहरुख खान", "अमिताभ बच्चन"]
result = pranaam.estimate_muslim_name_pattern(hindi_names, lang="hin")
```

Pandas Series retain their order:

```python
import pandas as pd

people = pd.DataFrame({"name": ["Shah Rukh Khan", "Amitabh Bachchan"]})
estimates = pranaam.estimate_muslim_name_pattern(people["name"])
people = pd.concat(
    [
        people,
        estimates[["muslim_score", "abstained"]],
    ],
    axis=1,
)
```

The command-line interface accepts the same language and refresh options:

```bash
pranaam --input "Shah Rukh Khan" --lang eng
pranaam --input "शाहरुख खान" --lang hin
```

Set `refresh_pinned=True` in Python or pass `--refresh-pinned` on the command
line to reload and verify a language's pinned files, even when that model is
already in memory. Pranaam redownloads files from the pinned Hugging Face
revision. When `PRANAAM_MODEL_DIR` is set, it instead rereads and verifies the
local files without downloading or replacing them. Neither mode follows a
mutable branch or switches model versions.

Pranaam withholds nothing it can compute: every supported name gets its
calibrated score, however mid-range. A name written outside the selected
model's supported script abstains and has no score, as does a name whose
normalized UTF-8 encoding exceeds `model_max_name_bytes`, and a blank or
non-text cell. These are name-pattern estimates for aggregate research;
never use them to label a person or make a consequential decision.
