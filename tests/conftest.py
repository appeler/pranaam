"""Shared test isolation for pranaam."""

from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def reset_naam_models() -> Generator[None, None, None]:
    """Keep class-level model caches isolated between tests."""
    from pranaam.naam import Naam

    original = Naam._models.copy()
    Naam._models.clear()
    yield
    Naam._models.clear()
    Naam._models.update(original)
