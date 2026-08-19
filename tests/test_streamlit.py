"""Tests for the optional Streamlit interface."""

import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).parents[1]


def test_streamlit_requirements_target_repository_root() -> None:
    """The deployment requirements resolve the editable project from repo root."""
    requirement = (PROJECT_ROOT / "streamlit" / "requirements.txt").read_text().strip()
    editable, target = requirement.split(maxsplit=1)

    assert editable == "-e"
    assert (PROJECT_ROOT / target.split("[", 1)[0]).resolve() == PROJECT_ROOT


def test_manual_app_flow_preserves_mixed_input_and_exposes_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rendered manual flow accepts mixed separators and shows results."""
    prediction = pd.DataFrame(
        {
            "name": ["Shah Rukh Khan", "Amitabh Bachchan", "Shah Rukh Khan"],
            "muslim_score": [0.95, 0.1, 0.95],
            "scored": [True, True, True],
            "abstained": [False, False, False],
            "abstention_reason": [None, None, None],
            "script_supported": [True, True, True],
            "normalized_utf8_bytes": [14, 17, 14],
            "reference_population": ["SEPRI household heads"] * 3,
            "label_source": ["test labels"] * 3,
            "calibration_reference": ["SEPRI household heads"] * 3,
            "model_language": ["eng"] * 3,
            "model_metadata_schema": [2, 2, 2],
            "model_version": ["3.0", "3.0", "3.0"],
            "model_revision": ["revision", "revision", "revision"],
            "model_max_name_bytes": [126, 126, 126],
        }
    )
    predict = Mock(return_value=prediction)
    monkeypatch.setattr("pranaam.estimate_muslim_name_pattern", predict)

    app = AppTest.from_file(PROJECT_ROOT / "streamlit" / "streamlit_app.py").run()
    app.text_area[0].input("Shah Rukh Khan, Amitabh Bachchan\nShah Rukh Khan").run()
    app.button[0].click().run()

    assert not app.exception
    assert app.title[0].value == "🔮 Pranaam: Muslim name-pattern estimates"
    assert "sensitive personal information" in app.warning[0].value
    pd.testing.assert_frame_equal(
        app.dataframe[0].value,
        prediction,
        check_dtype=False,
    )
    assert len(app.get("download_button")) == 1
    predict.assert_called_once_with(
        ["Shah Rukh Khan", "Amitabh Bachchan", "Shah Rukh Khan"], lang="eng"
    )


def test_download_file_uses_native_control(monkeypatch: pytest.MonkeyPatch) -> None:
    """The app exposes UTF-8 CSV bytes through the native download control."""
    namespace = runpy.run_path(PROJECT_ROOT / "streamlit" / "streamlit_app.py")
    download_button = Mock()
    monkeypatch.setattr(namespace["st"], "download_button", download_button)

    namespace["download_file"](pd.DataFrame({"name": ["Asha"]}))

    assert download_button.call_args.kwargs == {
        "label": "Download results as CSV",
        "data": b"name\nAsha\n",
        "file_name": "pranaam-results.csv",
        "mime": "text/csv",
    }


def test_parse_names_supports_mixed_separators() -> None:
    """Manual entry handles commas and newlines in the same submission."""
    namespace = runpy.run_path(PROJECT_ROOT / "streamlit" / "streamlit_app.py")

    assert namespace["parse_names"]("Asha, Ravi\nFatima,\n") == [
        "Asha",
        "Ravi",
        "Fatima",
    ]


def test_predict_dataframe_preserves_duplicates_and_missing_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CSV predictions preserve row count, order, duplicates, and missing names."""
    namespace = runpy.run_path(PROJECT_ROOT / "streamlit" / "streamlit_app.py")
    prediction = pd.DataFrame(
        {
            "name": ["Asha", "Asha"],
            "muslim_score": [0.05, 0.05],
            "scored": [True, True],
            "abstained": [False, False],
            "abstention_reason": [pd.NA, pd.NA],
            "script_supported": [True, True],
            "normalized_utf8_bytes": [4, 4],
            "reference_population": ["SEPRI household heads"] * 2,
            "label_source": ["test labels"] * 2,
            "calibration_reference": ["SEPRI household heads"] * 2,
            "model_language": ["eng"] * 2,
            "model_metadata_schema": [2, 2],
            "model_version": ["3.0", "3.0"],
            "model_revision": ["revision", "revision"],
            "model_max_name_bytes": [126, 126],
        }
    )
    predict = Mock(return_value=prediction)
    monkeypatch.setattr(namespace["pranaam"], "estimate_muslim_name_pattern", predict)
    source = pd.DataFrame({"name": ["Asha", None, "Asha"], "value": [1, 2, 3]})

    result = namespace["predict_dataframe"](source, "name", "eng")

    assert len(result) == len(source)
    assert result["value"].tolist() == [1, 2, 3]
    assert result["muslim_score"].tolist() == [0.05, pd.NA, 0.05]
    predict.assert_called_once_with(["Asha", "Asha"], lang="eng")


def test_predict_dataframe_rejects_non_text_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CSV input reports the first non-text row instead of coercing it."""
    namespace = runpy.run_path(PROJECT_ROOT / "streamlit" / "streamlit_app.py")
    source = pd.DataFrame({"name": ["Asha", 123]})

    with pytest.raises(TypeError, match="row 1"):
        namespace["predict_dataframe"](source, "name", "eng")


def test_launcher_runs_repo_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """The launcher invokes Streamlit with the repository app path."""
    namespace = runpy.run_path(PROJECT_ROOT / "streamlit" / "run_app.py")
    run = Mock()
    monkeypatch.setattr(subprocess, "run", run)

    namespace["main"]()

    command = run.call_args.args[0]
    assert command[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert command[4] == str(PROJECT_ROOT / "streamlit" / "streamlit_app.py")
    assert run.call_args.kwargs == {"check": True}
