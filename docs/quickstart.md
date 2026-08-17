# Quick start

`pred_rel` accepts one name, a list of names, or a pandas Series. It returns one
row per name.

```python
import pranaam

names = ["Shah Rukh Khan", "Amitabh Bachchan", "Abdul Kalam"]
result = pranaam.pred_rel(names, lang="eng")
print(result)
```

The output columns are:

* `name`: the input name
* `name_pattern_estimate`: associated pattern or `uncertain`
* `muslim_score`: calibrated score from 0 to 1
* `abstained` and `abstention_reason`: whether and why no pattern was returned
* `script_supported`: whether the selected model supports every input letter
* `model_version` and `model_revision`: immutable model provenance

Pass `lang="hin"` for names written in Hindi:

```python
hindi_names = ["शाहरुख खान", "अमिताभ बच्चन"]
result = pranaam.pred_rel(hindi_names, lang="hin")
```

Pandas Series retain their order:

```python
import pandas as pd

people = pd.DataFrame({"name": ["Shah Rukh Khan", "Amitabh Bachchan"]})
predictions = pranaam.pred_rel(people["name"])
people = pd.concat(
    [
        people,
        predictions[
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

Set `latest=True` in Python or pass `--latest` on the command line to refresh
and verify the files from the package's pinned Hugging Face revision, even when
the requested language is already loaded. This does not follow a mutable branch
or silently switch model versions.

Scores strictly between 0.2 and 0.8 abstain by default. A name written outside
the selected model's supported script also abstains and has no score. These are
name-pattern estimates for aggregate research; never use them to label a person
or make a consequential decision.
