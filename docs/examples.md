# Examples

## Add estimates to a DataFrame

`estimate_muslim_name_pattern` returns rows in the same order as the input.
Assign the estimate columns by position so duplicate names remain duplicate
rows.

```python
import pandas as pd

from pranaam import estimate_muslim_name_pattern

people = pd.DataFrame(
    {"name": ["Shah Rukh Khan", "Amitabh Bachchan", "Shah Rukh Khan"]}
)
estimates = estimate_muslim_name_pattern(people["name"], lang="eng")
people[["name_pattern_estimate", "muslim_score", "abstained"]] = estimates[
    ["name_pattern_estimate", "muslim_score", "abstained"]
].to_numpy()
```

## Process a CSV file

```python
import pandas as pd

from pranaam import estimate_muslim_name_pattern

people = pd.read_csv("people.csv")
names = people["name"]
if not names.dropna().map(lambda value: isinstance(value, str)).all():
    raise TypeError("The name column contains a non-string value")

valid = names.notna() & names.str.strip().ne("")
estimates = estimate_muslim_name_pattern(names.loc[valid], lang="eng")
columns = ["name_pattern_estimate", "muslim_score", "abstained"]
people.loc[valid, columns] = estimates[
    columns
].to_numpy()
people.to_csv("people_with_estimates.csv", index=False)
```

`estimate_muslim_name_pattern` requires every input to be a nonempty string.
The CSV example keeps missing and blank names in the output with missing
estimate values.

These estimates describe patterns learned from land and survey names. They do
not verify any person's religion. Use them for aggregate research only,
validate them for the population being studied, and do not use them to label
individuals or make consequential decisions.
