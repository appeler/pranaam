# Welcome to pranaam's documentation!

**pranaam** is a Python package for predicting religion from names using machine learning models trained on Bihar Land Records data. The package supports both Hindi and English names and provides high accuracy predictions.

:::{toctree}
:maxdepth: 2
:caption: Contents:

installation
quickstart
api
examples
contributing
:::

## Overview

Pranaam uses machine learning models trained on 4M unique records from Bihar Land Records data to predict religion (currently Muslim/not-Muslim) from names. The package supports:

* **High Accuracy**: 98% accuracy on unseen names for both Hindi and English
* **Multiple Languages**: Support for Hindi and English names
* **Easy to Use**: Simple API with pandas DataFrame output
* **Pre-trained Models**: Models are automatically downloaded and cached

## Quick Example

```python
from pranaam import pred_rel

# English names
names = ["Shah Rukh Khan", "Amitabh Bachchan"]
result = pred_rel(names)
print(result)

# Hindi names  
hindi_names = ["शाहरुख खान", "अमिताभ बच्चन"]
result = pred_rel(hindi_names, lang="hin")
print(result)
```

## Installation

Install pranaam using pip:

```bash
pip install pranaam
```

For development:

```bash
pip install -e .[dev]
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`