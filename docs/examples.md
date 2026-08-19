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
people = estimate_muslim_name_pattern(people, "name", lang="eng")
```

## Process a CSV file

```python
import pandas as pd

from pranaam import estimate_muslim_name_pattern

people = pd.read_csv("people.csv")
estimates = estimate_muslim_name_pattern(people, "name", lang="eng")
estimates.to_csv("people_with_estimates.csv", index=False)
```

Every input row survives to the output. Missing, blank, and non-text cells
abstain with `missing-name` and carry a missing score rather than raising or
dropping out of the frame, so the result always aligns with the source.

These estimates describe patterns learned from land and survey names. They do
not verify any person's religion. Use them for aggregate research only,
validate them for the population being studied, and do not use them to label
individuals or make consequential decisions.
