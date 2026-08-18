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
from pranaam import estimate_muslim_name_pattern

result = estimate_muslim_name_pattern(["Shah Rukh Khan", "Amitabh Bachchan"], lang="eng")
print(result)
```

The result contains a calibrated 0-to-1 score, an associated pattern or
abstention, explicit script and UTF-8 byte-limit support, the score's reference
and calibration populations, its label source, and immutable model provenance.
Pranaam downloads checksum-verified PyTorch model files from a pinned
[Hugging Face release](https://huggingface.co/gojiberries/pranaam) on first use
and stores them in the Hugging Face cache.

```python
hindi = estimate_muslim_name_pattern(["शाहरुख खान", "अमिताभ बच्चन"], lang="hin")
```

Use these estimates only for aggregate research with suitable validation. Do
not use them to label individuals or make consequential decisions.

## Indices

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
