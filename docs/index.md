# Pranaam documentation

Pranaam estimates whether a name follows Muslim or non-Muslim patterns in its
training data. The English and Hindi models were trained on Bihar land records.
The result is a model estimate, not a verified statement about a person.

```{toctree}
:maxdepth: 2

installation
quickstart
api.rst
examples
```

## Basic use

```python
from pranaam import pred_rel

result = pred_rel(["Shah Rukh Khan", "Amitabh Bachchan"], lang="eng")
print(result)
```

The result contains the original name, the predicted class, and the model's
Muslim probability on a 0 to 100 scale. Pranaam downloads checksum-verified
PyTorch model files from a pinned
[Hugging Face release](https://huggingface.co/gojiberries/pranaam) on first use
and stores them in the Hugging Face cache.

```python
hindi = pred_rel(["शाहरुख खान", "अमिताभ बच्चन"], lang="hin")
```

Use these estimates for aggregate research with suitable validation. Do not use
them to make decisions about individuals.

## Indices

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
