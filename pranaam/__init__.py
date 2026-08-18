"""Pranaam - calibrated name-pattern estimates.

The estimates describe statistical patterns in training data and are not
statements about a person's identity.
"""

from importlib.metadata import PackageNotFoundError, version

from .naam import Naam

estimate_muslim_name_pattern = Naam.estimate_muslim_name_pattern

try:
    __version__ = version("pranaam")
except PackageNotFoundError:
    __version__ = "0+unknown"
__all__ = ["Naam", "estimate_muslim_name_pattern"]
