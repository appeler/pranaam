"""Pranaam - calibrated name-pattern estimates.

The estimates describe statistical patterns in training data and are not
statements about a person's identity.
"""

from importlib.metadata import PackageNotFoundError, version

from .naam import Naam

pred_rel = Naam.pred_rel

try:
    __version__ = version("pranaam")
except PackageNotFoundError:
    __version__ = "0+unknown"
__all__ = ["Naam", "pred_rel"]
