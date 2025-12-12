"""Tests for utils module."""

import os
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest
import requests

from pranaam.utils import REPO_BASE_URL, _safe_extract_tar, download_file


class TestDownloadFile:
    """Test download_file function."""

    @patch("pranaam.utils.requests.Session")
    @patch("pranaam.utils._safe_extract_tar")
    @patch("pranaam.utils.Path")
    @patch("pranaam.utils.tqdm")
    def test_successful_download(
        self,
        mock_tqdm: Mock,
        mock_path: Mock,
        mock_extract: Mock,
        mock_session: Mock,
    ) -> None:
        """Test successful file download and extraction."""
        # Setup mocks
        mock_response = Mock()
        mock_response.headers = {"Content-Length": "1000"}
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_response.raise_for_status.return_value = None

        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session.return_value.__enter__.return_value = mock_session_instance

        # Mock tqdm context manager
        mock_tqdm.return_value.__enter__.return_value.update = Mock()

        # Mock Path operations
        mock_file_path = MagicMock()
        mock_target_path = MagicMock()
        mock_path.side_effect = [mock_target_path, mock_file_path]
        mock_target_path.__truediv__.return_value = mock_file_path
        mock_file_path.open.return_value.__enter__.return_value = Mock()
        mock_file_path.open.return_value.__exit__.return_value = None
        mock_file_path.unlink.return_value = None

        # Mock the extract function to not do anything
        mock_extract.return_value = None

        # Call function
        result = download_file("http://test.com", "/tmp/target", "test_file")

        # Verify
        assert result is True
        mock_session_instance.get.assert_called_once_with(
            REPO_BASE_URL, stream=True, allow_redirects=True, timeout=120
        )

    @patch("pranaam.utils.requests.Session")
    def test_network_error(self, mock_session: Mock) -> None:
        """Test handling of network errors."""
        mock_session_instance = Mock()
        mock_session_instance.get.side_effect = requests.exceptions.ConnectionError(
            "Network error"
        )
        mock_session.return_value.__enter__.return_value = mock_session_instance

        result = download_file("http://test.com", "/tmp/target", "test_file")

        assert result is False

    @patch("pranaam.utils.requests.Session")
    def test_http_error(self, mock_session: Mock) -> None:
        """Test handling of HTTP errors."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404 Not Found"
        )

        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session.return_value.__enter__.return_value = mock_session_instance

        result = download_file("http://test.com", "/tmp/target", "test_file")

        assert result is False

    @patch("pranaam.utils.requests.Session")
    @patch("pranaam.utils._safe_extract_tar")
    @patch("builtins.open", new_callable=mock_open)
    def test_extraction_error(
        self, mock_file: Mock, mock_extract: Mock, mock_session: Mock
    ) -> None:
        """Test handling of extraction errors."""
        # Setup successful download but failed extraction
        mock_response = Mock()
        mock_response.headers = {"Content-Length": "1000"}
        mock_response.iter_content.return_value = [b"chunk"]
        mock_response.raise_for_status.return_value = None

        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session.return_value.__enter__.return_value = mock_session_instance

        mock_extract.side_effect = tarfile.TarError("Corrupted tar file")

        result = download_file("http://test.com", "/tmp/target", "test_file")

        assert result is False

    @patch("pranaam.utils.requests.Session")
    @patch("pranaam.utils._safe_extract_tar")
    @patch("pranaam.utils.Path")
    @patch("pranaam.utils.tqdm")
    def test_no_content_length(
        self, mock_tqdm: Mock, mock_path: Mock, mock_extract: Mock, mock_session: Mock
    ) -> None:
        """Test handling when Content-Length header is missing."""
        mock_response = Mock()
        mock_response.headers = {}  # No Content-Length
        mock_response.iter_content.return_value = [b"chunk"]
        mock_response.raise_for_status.return_value = None

        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session.return_value.__enter__.return_value = mock_session_instance

        # Mock tqdm context manager
        mock_tqdm.return_value.__enter__.return_value.update = Mock()

        # Mock Path operations
        mock_file_path = MagicMock()
        mock_target_path = MagicMock()
        mock_path.side_effect = [mock_target_path, mock_file_path]
        mock_target_path.__truediv__.return_value = mock_file_path
        mock_file_path.open.return_value.__enter__.return_value = Mock()
        mock_file_path.open.return_value.__exit__.return_value = None
        mock_file_path.unlink.return_value = None

        # Mock the extract function to not do anything
        mock_extract.return_value = None

        result = download_file("http://test.com", "/tmp/target", "test_file")

        assert result is True


class TestSafeExtractTar:
    """Test _safe_extract_tar function."""

    def test_safe_extraction(self) -> None:
        """Test safe extraction of tar file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test tar file
            tar_path = os.path.join(temp_dir, "test.tar.gz")
            test_file_path = os.path.join(temp_dir, "test.txt")

            # Create test content
            with open(test_file_path, "w") as f:
                f.write("test content")

            # Create tar file
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(test_file_path, arcname="test.txt")

            # Remove original file
            os.remove(test_file_path)

            # Extract using our function
            extract_dir = os.path.join(temp_dir, "extracted")
            os.makedirs(extract_dir)

            _safe_extract_tar(Path(tar_path), Path(extract_dir))

            # Verify extraction
            extracted_file = os.path.join(extract_dir, "test.txt")
            assert os.path.exists(extracted_file)

            with open(extracted_file) as f:
                assert f.read() == "test content"

    def test_path_traversal_prevention(self) -> None:
        """Test prevention of path traversal attacks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tar_path = os.path.join(temp_dir, "malicious.tar.gz")

            # Create a tar file with path traversal attempt
            with tarfile.open(tar_path, "w:gz") as tar:
                # Create a TarInfo object with malicious path
                info = tarfile.TarInfo(name="../../../malicious.txt")
                content = b"malicious content"
                info.size = len(content)
                # Create a temporary file with the actual content
                import io

                tar.addfile(info, fileobj=io.BytesIO(content))

            extract_dir = os.path.join(temp_dir, "extracted")
            os.makedirs(extract_dir)

            # Should raise exception
            with pytest.raises(Exception, match="Attempted path traversal"):
                _safe_extract_tar(Path(tar_path), Path(extract_dir))

    def test_corrupted_tar_file(self) -> None:
        """Test handling of corrupted tar files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a corrupted file (not a valid tar)
            tar_path = os.path.join(temp_dir, "corrupted.tar.gz")
            with open(tar_path, "wb") as f:
                f.write(b"not a tar file")

            extract_dir = os.path.join(temp_dir, "extracted")
            os.makedirs(extract_dir)

            with pytest.raises(tarfile.TarError):
                _safe_extract_tar(Path(tar_path), Path(extract_dir))

    def test_nonexistent_tar_file(self) -> None:
        """Test handling of non-existent tar file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tar_path = os.path.join(temp_dir, "nonexistent.tar.gz")
            extract_dir = os.path.join(temp_dir, "extracted")
            os.makedirs(extract_dir)

            with pytest.raises((FileNotFoundError, tarfile.TarError)):
                _safe_extract_tar(Path(tar_path), Path(extract_dir))


class TestConstants:
    """Test module constants."""

    def test_repo_base_url_default(self) -> None:
        """Test default repository base URL."""
        # Should have default Harvard Dataverse URL
        assert "dataverse.harvard.edu" in REPO_BASE_URL

    @patch.dict(os.environ, {"PRANAAM_MODEL_URL": "http://custom.url/model"})
    def test_repo_base_url_custom(self) -> None:
        """Test custom repository URL from environment."""
        # Need to reimport to pick up environment variable
        import importlib

        import pranaam.utils

        importlib.reload(pranaam.utils)

        assert pranaam.utils.REPO_BASE_URL == "http://custom.url/model"


class TestErrorLogging:
    """Test error logging in utils functions."""

    @patch("pranaam.utils.logger")
    @patch("pranaam.utils.requests.Session")
    def test_network_error_logging(self, mock_session: Mock, mock_logger: Mock) -> None:
        """Test that network errors are logged properly."""
        mock_session_instance = Mock()
        mock_session_instance.get.side_effect = requests.exceptions.ConnectionError(
            "Network error"
        )
        mock_session.return_value.__enter__.return_value = mock_session_instance

        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            download_file("http://test.com", temp_dir, "test")

        mock_logger.error.assert_called()
        args, kwargs = mock_logger.error.call_args
        assert "Network error downloading models" in args[0]

    @patch("pranaam.utils.logger")
    @patch("pranaam.utils.requests.Session")
    @patch("pranaam.utils._safe_extract_tar")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pranaam.utils.tqdm")
    def test_extraction_error_logging(
        self,
        mock_tqdm: Mock,
        mock_file: Mock,
        mock_extract: Mock,
        mock_session: Mock,
        mock_logger: Mock,
    ) -> None:
        """Test that extraction errors are logged properly."""
        # Setup successful download
        mock_response = Mock()
        mock_response.headers = {"Content-Length": "1000"}
        mock_response.iter_content.return_value = [b"chunk"]
        mock_response.raise_for_status.return_value = None

        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session.return_value.__enter__.return_value = mock_session_instance

        # Mock tqdm context manager
        mock_tqdm.return_value.__enter__.return_value.update = Mock()

        # Setup extraction failure
        mock_extract.side_effect = tarfile.TarError("Extraction failed")

        download_file("http://test.com", "/tmp", "test")

        mock_logger.error.assert_called()
        args, kwargs = mock_logger.error.call_args
        assert "File extraction error" in args[0]
