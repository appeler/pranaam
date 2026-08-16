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
* `pred_label`: `muslim` or `not-muslim`
* `pred_prob_muslim`: the model probability on a 0 to 100 scale

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
    [people, predictions[["pred_label", "pred_prob_muslim"]]], axis=1
)
```

The command-line interface accepts the same language and refresh options:

```bash
predict_religion --input "Shah Rukh Khan" --lang eng
predict_religion --input "शाहरुख खान" --lang hin
```

Set `latest=True` in Python or pass `--latest` on the command line to refresh
and verify the files from the package's pinned Hugging Face revision, even when
the requested language is already loaded. This does not follow a mutable branch
or silently switch model versions.
