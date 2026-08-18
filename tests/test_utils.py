"""Tests for pinned and verified Hugging Face model downloads."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING
from unittest.mock import Mock, call, patch

import pytest

from pranaam.utils import (
    MODEL_REPO_ID,
    MODEL_REVISION,
    ModelDownloadError,
    ModelIntegrityError,
    download_model_file,
    file_sha256,
)

if TYPE_CHECKING:
    from pathlib import Path

FILENAME = "eng/model.safetensors"
PAYLOAD = b"verified model"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def test_file_sha256_streams_file(tmp_path: Path) -> None:
    """The digest helper returns the expected hexadecimal checksum."""
    path = tmp_path / "model"
    path.write_bytes(PAYLOAD)

    assert file_sha256(path) == DIGEST


@patch("pranaam.utils.hf_hub_download")
def test_download_uses_pinned_repo_and_revision(
    mock_download: Mock, tmp_path: Path
) -> None:
    """Hub resolution is immutable and the returned bytes are verified."""
    path = tmp_path / "model.safetensors"
    path.write_bytes(PAYLOAD)
    mock_download.return_value = str(path)

    with patch.dict("pranaam.utils.MODEL_SHA256", {FILENAME: DIGEST}, clear=True):
        assert download_model_file(FILENAME) == path

    mock_download.assert_called_once_with(
        repo_id=MODEL_REPO_ID,
        filename=FILENAME,
        revision=MODEL_REVISION,
        force_download=False,
        local_files_only=False,
    )


@patch("pranaam.utils.hf_hub_download")
def test_corrupt_cache_is_refetched_once(mock_download: Mock, tmp_path: Path) -> None:
    """A corrupt cached blob is replaced by one forced Hub download."""
    corrupt = tmp_path / "corrupt"
    correct = tmp_path / "correct"
    corrupt.write_bytes(b"corrupt")
    correct.write_bytes(PAYLOAD)
    mock_download.side_effect = [str(corrupt), str(correct)]

    with patch.dict("pranaam.utils.MODEL_SHA256", {FILENAME: DIGEST}, clear=True):
        assert download_model_file(FILENAME) == correct

    assert mock_download.call_args_list == [
        call(
            repo_id=MODEL_REPO_ID,
            filename=FILENAME,
            revision=MODEL_REVISION,
            force_download=False,
            local_files_only=False,
        ),
        call(
            repo_id=MODEL_REPO_ID,
            filename=FILENAME,
            revision=MODEL_REVISION,
            force_download=True,
            local_files_only=False,
        ),
    ]


@patch("pranaam.utils.hf_hub_download")
def test_forced_corrupt_download_fails_closed(
    mock_download: Mock, tmp_path: Path
) -> None:
    """An explicitly refreshed corrupt artifact is never accepted."""
    path = tmp_path / "corrupt"
    path.write_bytes(b"corrupt")
    mock_download.return_value = str(path)

    with (
        patch.dict("pranaam.utils.MODEL_SHA256", {FILENAME: DIGEST}, clear=True),
        pytest.raises(ModelIntegrityError, match="Checksum mismatch"),
    ):
        download_model_file(FILENAME, force_download=True)

    assert mock_download.call_count == 1


@patch("pranaam.utils.hf_hub_download")
def test_local_mirror_is_verified_without_hub(
    mock_download: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit local mirror supports deterministic offline operation."""
    path = tmp_path / FILENAME
    path.parent.mkdir(parents=True)
    path.write_bytes(PAYLOAD)
    monkeypatch.setenv("PRANAAM_MODEL_DIR", str(tmp_path))

    with patch.dict("pranaam.utils.MODEL_SHA256", {FILENAME: DIGEST}, clear=True):
        assert download_model_file(FILENAME) == path

    mock_download.assert_not_called()


@patch("pranaam.utils.hf_hub_download")
def test_refresh_rereads_local_mirror_without_downloading(
    mock_download: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Refreshing a local mirror verifies its file in place."""
    path = tmp_path / FILENAME
    path.parent.mkdir(parents=True)
    path.write_bytes(PAYLOAD)
    monkeypatch.setenv("PRANAAM_MODEL_DIR", str(tmp_path))

    with patch.dict("pranaam.utils.MODEL_SHA256", {FILENAME: DIGEST}, clear=True):
        assert download_model_file(FILENAME, force_download=True) == path

    mock_download.assert_not_called()


def test_local_mirror_checksum_mismatch_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A local override cannot bypass the release manifest."""
    path = tmp_path / FILENAME
    path.parent.mkdir(parents=True)
    path.write_bytes(b"wrong")
    monkeypatch.setenv("PRANAAM_MODEL_DIR", str(tmp_path))

    with (
        patch.dict("pranaam.utils.MODEL_SHA256", {FILENAME: DIGEST}, clear=True),
        pytest.raises(ModelIntegrityError, match="Checksum mismatch"),
    ):
        download_model_file(FILENAME)


@patch("pranaam.utils.hf_hub_download", side_effect=OSError("offline"))
def test_hub_failure_has_artifact_context(mock_download: Mock) -> None:
    """Network and cache failures name the exact pinned artifact."""
    with (
        patch.dict("pranaam.utils.MODEL_SHA256", {FILENAME: DIGEST}, clear=True),
        pytest.raises(ModelDownloadError, match=FILENAME),
    ):
        download_model_file(FILENAME, local_files_only=True)
    mock_download.assert_called_once()


def test_unknown_artifact_is_rejected_before_network() -> None:
    """Callers cannot fetch arbitrary files through the model helper."""
    with pytest.raises(ValueError, match="Unknown model artifact"):
        download_model_file("README.md")
