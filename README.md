# pranaam: predict religion from name

[![image](https://github.com/appeler/pranaam/workflows/test/badge.svg)](https://github.com/appeler/pranaam/actions?query=workflow%3Atest)
[![image](https://img.shields.io/pypi/v/pranaam.svg)](https://pypi.python.org/pypi/pranaam)
[![Documentation Status](https://readthedocs.org/projects/pranaam/badge/?version=latest)](http://pranaam.readthedocs.io/en/latest/?badge=latest)
[![image](https://static.pepy.tech/badge/pranaam)](https://pepy.tech/project/pranaam)

Pranaam uses the Bihar Land Records data, plot-level land records (N=
41.87 million plots or 12.13 individuals/accounts across 35,626
villages), to build machine learning models that predict religion and
caste from the name. Our final dataset has around 4M unique records. To
learn how to transform the data and the models underlying the package,
check the
[notebooks](https://github.com/appeler/pranaam/tree/main/pranaam/notebooks).

The first function we are releasing with the package is
[pred_rel]{.title-ref}, which predicts religion based on the name
(currently only [muslim]{.title-ref} or [not]{.title-ref}). (For
context, nearly 95% of India\'s population are Hindu or Muslim, with
Sikhs, Buddhists, Christians, and other groups making up the rest.) The
OOS accuracy assessed on unseen names is nearly 98% for both
[Hindi](https://github.com/appeler/pranaam_dev/blob/main/pranaam/notebooks/05_train_hindi.ipynb)
and
[English](https://github.com/appeler/pranaam_dev/blob/main/pranaam/notebooks/04_train_english.ipynb)
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

We strongly recommend installing [pranaam]{.title-ref} inside a Python
virtual environment. (see [venv
documentation](https://docs.python.org/3/library/venv.html#creating-virtual-environments))

    pip install pranaam

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
- [appeler/namesexdata](https://github.com/appeler/namesexdata) — Data on international first names and sex of people with that name
- [appeler/naamkaran](https://github.com/appeler/naamkaran) — generative model for names
- [appeler/outkast](https://github.com/appeler/outkast) — Using data from over 140M+ Indians from the SECC 2011, we map last names to caste (SC, ST, Other)
- [appeler/parsernaam](https://github.com/appeler/parsernaam) — AI name parsing. Predict first or last name using a DL model.

## Contributor Code of Conduct

The project welcomes contributions from everyone! It depends on it. To
maintain this welcoming atmosphere and to collaborate in a fun and
productive way, we expect contributors to the project to abide by the
[Contributor Code of
Conduct](http://contributor-covenant.org/version/1/0/0/).

## License

The package is released under the [MIT
License](https://opensource.org/licenses/MIT).
