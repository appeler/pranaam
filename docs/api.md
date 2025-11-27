# API Reference

This page contains the complete API documentation for pranaam.

## Main Functions

:::{automodule} pranaam.pranaam
:members:
:undoc-members:
:show-inheritance:
:::

## Core Classes

### Naam Class

:::{automodule} pranaam.naam
:members:
:undoc-members:
:show-inheritance:
:::

### Base Class

:::{automodule} pranaam.base
:members:
:undoc-members:
:show-inheritance:
:::

## Utility Functions

:::{automodule} pranaam.utils
:members:
:undoc-members:
:show-inheritance:
:::

## Logging Configuration

:::{automodule} pranaam.logging
:members:
:undoc-members:
:show-inheritance:
:::

## Function Parameters

### pred_rel

:::{py:function} pred_rel(input, lang="eng", latest=False)

Predict religion (Muslim/not-Muslim) from names.

:param input: Name(s) to predict religion for
:type input: str, list of str, or pandas.Series
:param lang: Language of the input names ("eng" for English, "hin" for Hindi)
:type lang: str, optional
:param latest: Whether to download the latest model if available
:type latest: bool, optional
:returns: DataFrame with columns ['name', 'pred_label', 'pred_prob_muslim']
:rtype: pandas.DataFrame
:raises ValueError: If invalid language is specified
:raises FileNotFoundError: If model files cannot be found or downloaded

**Examples:**

```python
# Single name
result = pred_rel("Shah Rukh Khan")

# Multiple names
result = pred_rel(["Shah Rukh Khan", "Amitabh Bachchan"], lang="eng")

# Hindi names
result = pred_rel(["शाहरुख खान"], lang="hin")

# Pandas Series
import pandas as pd
df = pd.DataFrame({"names": ["Shah Rukh Khan", "Amitabh Bachchan"]})
result = pred_rel(df["names"])
```
:::

### Return Values

The `pred_rel` function returns a pandas DataFrame with the following structure:

| Column | Type | Description |
|--------|------|-------------|
| name | str | Original input name |
| pred_label | str | Predicted religion ('muslim' or 'not-muslim') |
| pred_prob_muslim | float | Probability score (0-100) that person is Muslim |

## Model Information

The package uses two TensorFlow models:

* **English Model**: Trained on transliterated names from Hindi to English
* **Hindi Model**: Trained on original Hindi names from Bihar Land Records

Both models:

* Use SavedModel format (TensorFlow 2.14.1 compatible)
* Achieve 98% out-of-sample accuracy
* Are automatically downloaded and cached (306MB total)
* Use character-level and n-gram features

## Exception Handling

Common exceptions that may be raised:

:::{py:exception} ValueError

Raised when invalid parameters are provided (e.g., unsupported language).
:::

:::{py:exception} FileNotFoundError

Raised when model files cannot be found or downloaded.
:::

:::{py:exception} ImportError

Raised when required dependencies are missing.
:::

## Type Hints

The package includes comprehensive type annotations for all public functions:

```python
from typing import Union, List
import pandas as pd

def pred_rel(
    input: Union[str, List[str], pd.Series],
    lang: str = "eng",
    latest: bool = False
) -> pd.DataFrame:
    ...
```

## Constants

:::{py:data} SUPPORTED_LANGUAGES

List of supported language codes: ['eng', 'hin']
:::

:::{py:data} MODEL_URLS

Dictionary mapping model names to their download URLs
:::

:::{py:data} DEFAULT_CACHE_DIR

Default directory for caching downloaded models
:::