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


def test_manual_app_flow_preserves_mixed_input_and_exposes_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rendered manual flow accepts mixed separators and shows results."""
    prediction = pd.DataFrame(
        {
            "name": ["Shah Rukh Khan", "Amitabh Bachchan", "Shah Rukh Khan"],
            "pred_label": ["muslim", "not-muslim", "muslim"],
            "pred_prob_muslim": [95.0, 10.0, 95.0],
        }
    )
    predict = Mock(return_value=prediction)
    monkeypatch.setattr("pranaam.pred_rel", predict)

    app = AppTest.from_file(PROJECT_ROOT / "streamlit" / "streamlit_app.py").run()
    app.text_area[0].input("Shah Rukh Khan, Amitabh Bachchan\nShah Rukh Khan").run()
    app.button[0].click().run()

    assert not app.exception
    assert app.title[0].value == "🔮 Pranaam: name-pattern classification"
    assert "sensitive personal information" in app.warning[0].value
    assert app.dataframe[0].value.equals(prediction)
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
            "pred_label": ["not-muslim", "not-muslim"],
            "pred_prob_muslim": [5.0, 5.0],
        }
    )
    predict = Mock(return_value=prediction)
    monkeypatch.setattr(namespace["pranaam"], "pred_rel", predict)
    source = pd.DataFrame({"name": ["Asha", None, "Asha"], "value": [1, 2, 3]})

    result = namespace["predict_dataframe"](source, "name", "eng")

    assert len(result) == len(source)
    assert result["value"].tolist() == [1, 2, 3]
    assert result["pred_label"].tolist() == ["not-muslim", pd.NA, "not-muslim"]
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
