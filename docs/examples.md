# Examples

## Add predictions to a DataFrame

`pred_rel` returns rows in the same order as the input. Assign the prediction
columns by position so duplicate names remain duplicate rows.

```python
import pandas as pd

from pranaam import pred_rel

people = pd.DataFrame(
    {"name": ["Shah Rukh Khan", "Amitabh Bachchan", "Shah Rukh Khan"]}
)
predictions = pred_rel(people["name"], lang="eng")
people[["pred_label", "pred_prob_muslim"]] = predictions[
    ["pred_label", "pred_prob_muslim"]
].to_numpy()
```

## Process a CSV file

```python
import pandas as pd

from pranaam import pred_rel

people = pd.read_csv("people.csv")
names = people["name"]
if not names.dropna().map(lambda value: isinstance(value, str)).all():
    raise TypeError("The name column contains a non-string value")

valid = names.notna() & names.str.strip().ne("")
predictions = pred_rel(names.loc[valid], lang="eng")
people.loc[valid, ["pred_label", "pred_prob_muslim"]] = predictions[
    ["pred_label", "pred_prob_muslim"]
].to_numpy()
people.to_csv("people_with_predictions.csv", index=False)
```

`pred_rel` requires every input to be a nonempty string. The CSV example keeps
missing and blank names in the output with missing prediction values.

These estimates describe patterns learned from Bihar land-record names. They do
not verify any person's religion. Use them for aggregate research only, validate
them for the population being studied, and do not use them to make decisions
about individuals.
