# pranaam: predict religion from name

[![ci](https://github.com/appeler/pranaam/actions/workflows/ci.yml/badge.svg)](https://github.com/appeler/pranaam/actions/workflows/ci.yml)
[![image](https://img.shields.io/pypi/v/pranaam.svg)](https://pypi.python.org/pypi/pranaam)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://appeler.github.io/pranaam/)
[![image](https://static.pepy.tech/badge/pranaam)](https://pepy.tech/project/pranaam)

Pranaam uses the Bihar Land Records data, plot-level land records (N=
41.87 million plots or 12.13 individuals/accounts across 35,626
villages), to build machine learning models that predict religion and
caste from the name. Our final dataset has around 4M unique records. To
learn how to transform the data and the models underlying the package,
check the
[notebooks](https://github.com/appeler/pranaam/tree/main/).

The first function we are releasing with the package is
[pred_rel]{.title-ref}, which predicts religion based on the name
(currently only [muslim]{.title-ref} or [not]{.title-ref}). (For
context, nearly 95% of India\'s population are Hindu or Muslim, with
Sikhs, Buddhists, Christians, and other groups making up the rest.) The
OOS accuracy assessed on unseen names is nearly 98% for both
[Hindi](https://github.com/appeler/pranaam_dev/blob/main/05_train_hindi.ipynb)
and
[English](https://github.com/appeler/pranaam_dev/blob/main/04_train_english.ipynb)
models.

Our training data is in Hindi. To build models that classify names
provided in English, we used the
[indicate](https://github.com/in-rolls/indicate) package to
transliterate our training data to English.

We are releasing this software in the hope that it enables activists and
researchers

1)  Highlight biases
2)  Fight biases
3)  Prevent biases (regress out some of these biases in models built on
    natural language corpus with person names).

## Install

We strongly recommend installing pranaam inside a Python virtual environment. (see [venv documentation](https://docs.python.org/3/library/venv.html#creating-virtual-environments))

### Standard Installation

```bash
pip install pranaam
```

This installs TensorFlow 2.14.1, which is known to work correctly with the models.

### Requirements

- Python 3.10 or 3.11 (TensorFlow 2.14.1 compatibility requirement)
- TensorFlow 2.14.1 (automatically installed)

> **Note**: This package requires TensorFlow 2.14.1 with Keras 2.14.0 for model compatibility. Python 3.12+ is not currently supported due to TensorFlow availability constraints.

## General API

1.  pranaam.pred_rel takes a list of Hindi/English names and predicts
    whether the person is Muslim or not.

## Examples

By using names in English :

    from pranaam import pranaam
    names = ["Shah Rukh Khan", "Amitabh Bachchan"]
    result = pranaam.pred_rel(names)
    print(result)

output -:

    name  pred_label  pred_prob_muslim
    0    Shah Rukh Khan      muslim              73.0
    1  Amitabh Bachchan  not-muslim              27.0

By using names in Hindi :

    from pranaam import pranaam
    names = ["शाहरुख खान", "अमिताभ बच्चन"]
    result = pranaam.pred_rel(names, lang="hin")
    print(result)

output -:

    name  pred_label  pred_prob_muslim
    0    शाहरुख खान      muslim              73.0
    1  अमिताभ बच्चन  not-muslim              27.0

## Functions

We expose one function, which takes Hindi/English text (name) and
predicts religion and caste.

- **pranaam.pred_rel(input)**
  - What it does:
    - predicts religion based on hindi/english text (name)
  - Output
    - Returns pandas with name and label (muslim/not-muslim)

## Authors

Rajashekar Chintalapati, Aaditya Dar, and Gaurav Sood


## 🔗 Adjacent Repositories

- [appeler/naampy](https://github.com/appeler/naampy) — Infer Sociodemographic Characteristics from Names Using Indian Electoral Rolls
- [appeler/parsernaam](https://github.com/appeler/parsernaam) — AI name parsing. Predict first or last name using a DL model.
- [appeler/namesexdata](https://github.com/appeler/namesexdata) — Data on international first names and sex of people with that name
- [appeler/graphic_names](https://github.com/appeler/graphic_names) — Infer the gender of person with a particular first name using Google image search and Clarifai
- [appeler/ethnicolr2](https://github.com/appeler/ethnicolr2) — Ethnicolr implementation with new models in pytorch
## Contributor Code of Conduct

The project welcomes contributions from everyone! It depends on it. To
maintain this welcoming atmosphere and to collaborate in a fun and
productive way, we expect contributors to the project to abide by the
[Contributor Code of
Conduct](http://contributor-covenant.org/version/1/0/0/).

## License

The package is released under the [MIT
License](https://opensource.org/licenses/MIT).
