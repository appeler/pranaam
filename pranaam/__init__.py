"""Pranaam - Religion prediction from names.

A Python package for predicting religion from names using machine learning
models trained on Bihar Land Records data.
"""

from importlib.metadata import PackageNotFoundError, version

from .naam import Naam

pred_rel = Naam.pred_rel

try:
    __version__ = version("pranaam")
except PackageNotFoundError:
    __version__ = "0+unknown"
__all__ = ["Naam", "pred_rel"]
