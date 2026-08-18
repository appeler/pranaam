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
* `name_pattern_estimate`: associated pattern or `uncertain`
* `muslim_score`: calibrated score from 0 to 1
* `abstained` and `abstention_reason`: whether and why no pattern was returned
* `script_supported`: whether the selected model supports every input letter
* `normalized_utf8_bytes` and `input_truncated`: explicit byte-limit support
* `reference_population`, `label_source`, and `calibration_population`: the
  population and labeling scope of the score
* `model_language`, `model_metadata_schema`, `model_version`, `model_revision`,
  and `model_max_name_bytes`: model provenance and support boundary

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
        estimates[
            ["name_pattern_estimate", "muslim_score", "abstained"]
        ],
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

Scores strictly between 0.2 and 0.8 abstain by default. A name written outside
the selected model's supported script also abstains and has no score. Names
whose normalized UTF-8 encoding exceeds `model_max_name_bytes` likewise
abstain, with `input_truncated=True` and
`abstention_reason="input-truncated"`. These are name-pattern estimates for
aggregate research; never use them to label a person or make a consequential
decision.
