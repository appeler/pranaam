"""Pranaam - Religion prediction from names.

A Python package for predicting religion from names using machine learning
models trained on Bihar Land Records data.
"""

from .pranaam import pred_rel
from .naam import Naam

__version__ = "0.0.2"
__all__ = ["pred_rel", "Naam"]
