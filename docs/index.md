# Pranaam documentation

Pranaam returns calibrated estimates of whether a name follows Muslim- or
non-Muslim-associated patterns in its training data. The result is a
name-pattern estimate, not a statement about a person's identity.

```{toctree}
:maxdepth: 2

installation
quickstart
api.rst
examples
model_v3_audit
```

## Basic use

```python
from pranaam import pred_rel

result = pred_rel(["Shah Rukh Khan", "Amitabh Bachchan"], lang="eng")
print(result)
```

The result contains a calibrated 0-to-1 score, an associated pattern or
abstention, script-support status, and immutable model provenance. Pranaam
downloads checksum-verified PyTorch model files from a pinned
[Hugging Face release](https://huggingface.co/gojiberries/pranaam) on first use
and stores them in the Hugging Face cache.

```python
hindi = pred_rel(["शाहरुख खान", "अमिताभ बच्चन"], lang="hin")
```

Use these estimates only for aggregate research with suitable validation. Do
not use them to label individuals or make consequential decisions.

## Indices

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
