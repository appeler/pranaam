"""Tests for secure model downloading and extraction."""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
import requests

from pranaam.utils import (
    SecurityError,
    _install_verified_model,
    _safe_extract_tar,
    download_file,
    get_model_url,
)

if TYPE_CHECKING:
    from pathlib import Path

MODEL_URL = "https://example.test/model"


def _archive_bytes(files: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return stream.getvalue()


def _session_for(
    payload: bytes, *, status_code: int = 200, content_length: int | None = None
) -> Mock:
    response = Mock()
    response.status_code = status_code
    response.headers = {
        "Content-Length": str(
            len(payload) if content_length is None else content_length
        )
    }
    response.iter_content.return_value = [payload]
    response.raise_for_status.return_value = None
    session = Mock()
    session.get.return_value = response
    context = Mock()
    context.__enter__ = Mock(return_value=session)
    context.__exit__ = Mock(return_value=False)
    return context


def _checksums(english: bytes = b"english", hindi: bytes = b"hindi") -> dict[str, str]:
    return {
        "eng_model.keras": hashlib.sha256(english).hexdigest(),
        "hin_model.keras": hashlib.sha256(hindi).hexdigest(),
    }


class TestDownloadFile:
    """Test verified, atomic model installation."""

    @patch("pranaam.utils.requests.Session")
    def test_download_uses_argument_and_verifies_models(
        self, mock_session: Mock, tmp_path: Path
    ) -> None:
        """The requested URL is used and verified files are installed."""
        payload = _archive_bytes(
            {
                "bundle/eng_model.keras": b"english",
                "bundle/hin_model.keras": b"hindi",
            }
        )
        mock_session.return_value = _session_for(payload)

        with patch("pranaam.utils.MODEL_SHA256", _checksums()):
            assert download_file(MODEL_URL, str(tmp_path), "bundle")

        request = mock_session.return_value.__enter__.return_value.get
        request.assert_called_once_with(
            MODEL_URL,
            stream=True,
            allow_redirects=True,
            timeout=120,
        )
        assert (tmp_path / "bundle" / "eng_model.keras").read_bytes() == b"english"
        assert not (tmp_path / ".bundle.previous").exists()
        assert not list(tmp_path.glob(".bundle-*"))

    @patch("pranaam.utils.requests.Session")
    def test_verified_refresh_replaces_old_cache(
        self, mock_session: Mock, tmp_path: Path
    ) -> None:
        """A verified refresh replaces the previous bundle completely."""
        installed = tmp_path / "bundle"
        installed.mkdir()
        (installed / "old.txt").write_text("old")
        payload = _archive_bytes(
            {
                "bundle/eng_model.keras": b"english",
                "bundle/hin_model.keras": b"hindi",
            }
        )
        mock_session.return_value = _session_for(payload)

        with patch("pranaam.utils.MODEL_SHA256", _checksums()):
            assert download_file(MODEL_URL, str(tmp_path), "bundle")

        assert not (installed / "old.txt").exists()
        assert (installed / "hin_model.keras").read_bytes() == b"hindi"
        assert not (tmp_path / ".bundle.previous").exists()

    @patch("pranaam.utils.requests.Session")
    def test_checksum_mismatch_preserves_old_cache(
        self, mock_session: Mock, tmp_path: Path
    ) -> None:
        """Unverified model bytes never replace a known-good cache."""
        installed = tmp_path / "bundle"
        installed.mkdir()
        (installed / "known-good").write_bytes(b"old")
        payload = _archive_bytes(
            {
                "bundle/eng_model.keras": b"wrong",
                "bundle/hin_model.keras": b"wrong",
            }
        )
        mock_session.return_value = _session_for(payload)

        assert not download_file(MODEL_URL, str(tmp_path), "bundle")
        assert (installed / "known-good").read_bytes() == b"old"
        assert not list(tmp_path.glob(".bundle-*"))

    @patch("pranaam.utils.requests.Session")
    def test_truncated_response_preserves_old_cache(
        self, mock_session: Mock, tmp_path: Path
    ) -> None:
        """A short response cannot replace an installed bundle."""
        installed = tmp_path / "bundle"
        installed.mkdir()
        (installed / "known-good").write_bytes(b"old")
        payload = _archive_bytes(
            {
                "bundle/eng_model.keras": b"english",
                "bundle/hin_model.keras": b"hindi",
            }
        )
        mock_session.return_value = _session_for(
            payload, content_length=len(payload) + 100
        )

        with patch("pranaam.utils.MODEL_SHA256", _checksums()):
            assert not download_file(MODEL_URL, str(tmp_path), "bundle")

        assert (installed / "known-good").read_bytes() == b"old"

    @patch("pranaam.utils.requests.Session")
    def test_waf_challenge_is_not_a_download(
        self, mock_session: Mock, tmp_path: Path
    ) -> None:
        """An HTTP 202 challenge is a failure even though it is a 2xx response."""
        context = _session_for(b"", status_code=202)
        response = context.__enter__.return_value.get.return_value
        response.headers["x-amzn-waf-action"] = "challenge"
        mock_session.return_value = context

        assert not download_file(MODEL_URL, str(tmp_path), "bundle")
        assert not (tmp_path / "bundle").exists()

    @patch("pranaam.utils.requests.Session")
    def test_network_error_restores_interrupted_backup(
        self, mock_session: Mock, tmp_path: Path
    ) -> None:
        """A cache left mid-swap is restored before attempting a new download."""
        backup = tmp_path / ".bundle.previous"
        backup.mkdir()
        (backup / "known-good").write_bytes(b"old")
        session = Mock()
        session.get.side_effect = requests.ConnectionError("offline")
        context = Mock()
        context.__enter__ = Mock(return_value=session)
        context.__exit__ = Mock(return_value=False)
        mock_session.return_value = context

        assert not download_file(MODEL_URL, str(tmp_path), "bundle")
        assert (tmp_path / "bundle" / "known-good").read_bytes() == b"old"
        assert not backup.exists()

    def test_failed_directory_swap_restores_old_cache(self, tmp_path: Path) -> None:
        """An installation error rolls the previous directory back into place."""
        installed = tmp_path / "bundle"
        installed.mkdir()
        (installed / "known-good").write_bytes(b"old")
        extracted = tmp_path / "staging" / "bundle"
        extracted.mkdir(parents=True)
        (extracted / "new").write_bytes(b"new")
        real_replace = os.replace

        def fail_new_install(
            source: os.PathLike[str], destination: os.PathLike[str]
        ) -> None:
            if source == extracted and destination == installed:
                raise OSError("simulated swap failure")
            real_replace(source, destination)

        with (
            patch("pranaam.utils.os.replace", side_effect=fail_new_install),
            pytest.raises(OSError, match="simulated swap failure"),
        ):
            _install_verified_model(extracted, tmp_path, "bundle")

        assert (installed / "known-good").read_bytes() == b"old"
        assert not (tmp_path / ".bundle.previous").exists()

    def test_model_url_is_read_when_requested(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A runtime environment override selects a trusted mirror."""
        monkeypatch.setenv("PRANAAM_MODEL_URL", MODEL_URL)

        assert get_model_url() == MODEL_URL

    def test_empty_model_url_override_uses_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unset repository secret does not produce an invalid empty URL."""
        monkeypatch.setenv("PRANAAM_MODEL_URL", "")

        assert get_model_url().startswith("https://dataverse.harvard.edu/")


class TestSafeExtractTar:
    """Test archive extraction boundaries."""

    def test_safe_extraction(self, tmp_path: Path) -> None:
        """Ordinary data files extract successfully."""
        archive_path = tmp_path / "safe.tar.gz"
        archive_path.write_bytes(_archive_bytes({"model/file.txt": b"content"}))
        target = tmp_path / "target"

        _safe_extract_tar(archive_path, target)

        assert (target / "model" / "file.txt").read_bytes() == b"content"

    def test_path_traversal_is_rejected(self, tmp_path: Path) -> None:
        """Parent-directory paths cannot escape the destination."""
        archive_path = tmp_path / "traversal.tar.gz"
        archive_path.write_bytes(_archive_bytes({"../../../outside.txt": b"bad"}))

        with pytest.raises(SecurityError, match="path traversal"):
            _safe_extract_tar(archive_path, tmp_path / "target")

    def test_outside_symlink_is_rejected(self, tmp_path: Path) -> None:
        """The data filter rejects links outside the destination."""
        archive_path = tmp_path / "link.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            info = tarfile.TarInfo("link")
            info.type = tarfile.SYMTYPE
            info.linkname = "../../outside"
            archive.addfile(info)

        with pytest.raises(tarfile.FilterError):
            _safe_extract_tar(archive_path, tmp_path / "target")

    def test_corrupted_archive_is_rejected(self, tmp_path: Path) -> None:
        """Invalid archive bytes raise a tar error."""
        archive_path = tmp_path / "corrupt.tar.gz"
        archive_path.write_bytes(b"not a tar archive")

        with pytest.raises(tarfile.TarError):
            _safe_extract_tar(archive_path, tmp_path / "target")
