"""Test configuration and fixtures for pranaam tests."""

import os
import tempfile
from collections.abc import Generator
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def sample_english_names() -> list[str]:
    """Sample English names for testing."""
    return ["Shah Rukh Khan", "Amitabh Bachchan", "Rajesh Khanna", "Mohammed Ali"]


@pytest.fixture
def sample_hindi_names() -> list[str]:
    """Sample Hindi names for testing."""
    return ["शाहरुख खान", "अमिताभ बच्चन", "राजेश खन्ना", "मोहम्मद अली"]


@pytest.fixture
def expected_predictions() -> dict[str, dict[str, Any]]:
    """Expected predictions for sample names."""
    return {
        "Shah Rukh Khan": {"label": "muslim", "prob_range": (60, 90)},
        "Amitabh Bachchan": {"label": "not-muslim", "prob_range": (10, 40)},
        "Rajesh Khanna": {"label": "not-muslim", "prob_range": (10, 40)},
        "Mohammed Ali": {"label": "muslim", "prob_range": (70, 95)},
    }


@pytest.fixture
def mock_tensorflow_model() -> Mock:
    """Mock TensorFlow model for testing."""
    model = Mock()

    # Mock prediction results - returns probabilities for [not-muslim, muslim]
    def mock_predict(names: Any, verbose: int = 0) -> Any:
        results = []
        for name in names:
            if any(
                muslim_name in str(name).lower()
                for muslim_name in ["shah", "khan", "mohammed", "ali"]
            ):
                # Higher probability for muslim class
                results.append([0.2, 0.8])
            else:
                # Higher probability for not-muslim class
                results.append([0.8, 0.2])
        return np.array(results)

    model.predict = mock_predict
    return model


@pytest.fixture
def temp_model_dir() -> Generator[str, None, None]:
    """Temporary directory for model files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create model directory structure
        model_path = os.path.join(temp_dir, "eng_and_hindi_models_v1")
        os.makedirs(os.path.join(model_path, "eng_model"))
        os.makedirs(os.path.join(model_path, "hin_model"))
        yield temp_dir


@pytest.fixture
def mock_requests_get() -> Mock:
    """Mock requests.get for download testing."""
    mock_response = Mock()
    mock_response.headers = {"Content-Length": "1000"}
    mock_response.iter_content.return_value = [b"test data chunk"]
    mock_response.raise_for_status.return_value = None
    return mock_response


@pytest.fixture(autouse=True)
def reset_naam_class() -> Generator[None, None, None]:
    """Reset Naam class state between tests."""
    from pranaam.naam import Naam

    # Store original values
    original_weights_loaded = Naam.weights_loaded
    original_model = Naam.model
    original_cur_lang = Naam.cur_lang

    yield

    # Reset to original values
    Naam.weights_loaded = original_weights_loaded
    Naam.model = original_model
    Naam.cur_lang = original_cur_lang


@pytest.fixture
def caplog_debug(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Capture log messages at DEBUG level."""
    import logging

    caplog.set_level(logging.DEBUG)
    return caplog
