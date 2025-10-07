"""Tests for base module."""

from pathlib import Path
from unittest.mock import Mock, patch

from pranaam.base import Base


class TestBase:
    """Test Base class functionality."""

    def test_base_class_attributes(self) -> None:
        """Test Base class has expected attributes."""
        assert hasattr(Base, "MODELFN")
        assert hasattr(Base, "load_model_data")
        assert Base.MODELFN is None

    def test_load_model_data_no_modelfn(self) -> None:
        """Test load_model_data when MODELFN is None."""

        # Create a test class without MODELFN set
        class TestClass(Base):
            MODELFN = None

        result = TestClass.load_model_data("test_file")
        assert result is None

    @patch("pranaam.base.files")
    @patch("pranaam.base.download_file")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.mkdir")
    def test_load_model_data_success(
        self,
        mock_mkdir: Mock,
        mock_exists: Mock,
        mock_download: Mock,
        mock_files: Mock,
    ) -> None:
        """Test successful model data loading."""
        # Setup mocks - make files() return a mock that converts to a valid path string
        mock_package_dir = Mock()
        mock_package_dir.__str__.return_value = "/fake/package"
        mock_files.return_value = mock_package_dir

        # File doesn't exist (so download gets called)
        mock_exists.return_value = False
        mock_download.return_value = True

        # Create test class
        class TestClass(Base):
            MODELFN = "model"

        result = TestClass.load_model_data("test_model", latest=False)

        assert result == Path("/fake/package/model")
        mock_mkdir.assert_called_once_with(exist_ok=True)
        mock_download.assert_called_once()

    @patch("pranaam.base.files")
    @patch("pranaam.base.download_file")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.mkdir")
    def test_load_model_data_file_exists_no_latest(
        self,
        mock_mkdir: Mock,
        mock_exists: Mock,
        mock_download: Mock,
        mock_files: Mock,
    ) -> None:
        """Test model loading when file exists and latest=False."""
        # Setup mocks
        mock_package_dir = Mock()
        mock_package_dir.__str__.return_value = "/fake/package"
        mock_files.return_value = mock_package_dir

        # File exists
        mock_exists.return_value = True

        class TestClass(Base):
            MODELFN = "model"

        result = TestClass.load_model_data("test_model", latest=False)

        assert result == Path("/fake/package/model")
        # Should not download since file exists and latest=False
        mock_download.assert_not_called()

    @patch("pranaam.base.files")
    @patch("pranaam.base.download_file")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.mkdir")
    def test_load_model_data_force_latest(
        self,
        mock_mkdir: Mock,
        mock_exists: Mock,
        mock_download: Mock,
        mock_files: Mock,
    ) -> None:
        """Test model loading with latest=True forces redownload."""
        # Setup mocks
        mock_package_dir = Mock()
        mock_package_dir.__str__.return_value = "/fake/package"
        mock_files.return_value = mock_package_dir

        mock_exists.return_value = True  # File exists
        mock_download.return_value = True

        class TestClass(Base):
            MODELFN = "model"

        result = TestClass.load_model_data("test_model", latest=True)

        assert result == Path("/fake/package/model")
        # Should download even though file exists because latest=True
        mock_download.assert_called_once()

    @patch("pranaam.base.files")
    @patch("pranaam.base.download_file")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.mkdir")
    def test_load_model_data_download_failure(
        self,
        mock_mkdir: Mock,
        mock_exists: Mock,
        mock_download: Mock,
        mock_files: Mock,
    ) -> None:
        """Test handling of download failure."""
        # Setup mocks
        mock_package_dir = Mock()
        mock_package_dir.__str__.return_value = "/fake/package"
        mock_files.return_value = mock_package_dir

        # File doesn't exist
        mock_exists.return_value = False
        mock_download.return_value = False  # Download fails

        class TestClass(Base):
            MODELFN = "model"

        result = TestClass.load_model_data("test_model")

        # Should still return path even if download fails
        assert result == Path("/fake/package/model")

    @patch("pranaam.base.files")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.mkdir")
    def test_load_model_data_creates_directory(
        self, mock_mkdir: Mock, mock_exists: Mock, mock_files: Mock
    ) -> None:
        """Test that model directory is created if it doesn't exist."""
        # Setup mocks
        mock_package_dir = Mock()
        mock_package_dir.__str__.return_value = "/fake/package"
        mock_files.return_value = mock_package_dir

        mock_exists.return_value = False  # File doesn't exist

        class TestClass(Base):
            MODELFN = "model"

        with patch("pranaam.base.download_file", return_value=True):
            result = TestClass.load_model_data("test_model")

        mock_mkdir.assert_called_once_with(exist_ok=True)
        assert result == Path("/fake/package/model")


class TestBaseInheritance:
    """Test Base class inheritance patterns."""

    def test_subclass_can_override_modelfn(self) -> None:
        """Test that subclasses can override MODELFN."""

        class CustomBase(Base):
            MODELFN = "custom_model"

        assert CustomBase.MODELFN == "custom_model"
        assert Base.MODELFN is None  # Original unchanged

    def test_multiple_subclasses_independent(self) -> None:
        """Test that multiple subclasses have independent MODELFN values."""

        class BaseA(Base):
            MODELFN = "model_a"

        class BaseB(Base):
            MODELFN = "model_b"

        assert BaseA.MODELFN == "model_a"
        assert BaseB.MODELFN == "model_b"
        assert Base.MODELFN is None


class TestBaseLogging:
    """Test logging in Base class."""

    @patch("pranaam.base.logger")
    @patch("pranaam.base.files")
    @patch("pranaam.base.download_file")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.mkdir")
    def test_debug_logging_download(
        self,
        mock_mkdir: Mock,
        mock_exists: Mock,
        mock_download: Mock,
        mock_files: Mock,
        mock_logger: Mock,
    ) -> None:
        """Test debug logging during download."""
        # Setup mocks
        mock_package_dir = Mock()
        mock_package_dir.__str__.return_value = "/fake/package"
        mock_files.return_value = mock_package_dir

        # File doesn't exist
        mock_exists.return_value = False
        mock_download.return_value = True

        class TestClass(Base):
            MODELFN = "model"

        TestClass.load_model_data("test_model")

        # Should log download message
        mock_logger.debug.assert_called()
        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Downloading model data" in call for call in debug_calls)

    @patch("pranaam.base.logger")
    @patch("pranaam.base.files")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.mkdir")
    def test_debug_logging_existing_model(
        self, mock_mkdir: Mock, mock_exists: Mock, mock_files: Mock, mock_logger: Mock
    ) -> None:
        """Test debug logging when using existing model."""
        # Setup mocks
        mock_package_dir = Mock()
        mock_package_dir.__str__.return_value = "/fake/package"
        mock_files.return_value = mock_package_dir

        mock_exists.return_value = True  # File exists

        class TestClass(Base):
            MODELFN = "model"

        TestClass.load_model_data("test_model")

        # Should log using existing model message
        mock_logger.debug.assert_called()
        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Using model data from" in call for call in debug_calls)

    @patch("pranaam.base.logger")
    @patch("pranaam.base.files")
    @patch("pranaam.base.download_file")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.mkdir")
    def test_error_logging_download_failure(
        self,
        mock_mkdir: Mock,
        mock_exists: Mock,
        mock_download: Mock,
        mock_files: Mock,
        mock_logger: Mock,
    ) -> None:
        """Test error logging when download fails."""
        # Setup mocks
        mock_package_dir = Mock()
        mock_package_dir.__str__.return_value = "/fake/package"
        mock_files.return_value = mock_package_dir

        # File doesn't exist
        mock_exists.return_value = False
        mock_download.return_value = False  # Download fails

        class TestClass(Base):
            MODELFN = "model"

        TestClass.load_model_data("test_model")

        # Should log error message
        mock_logger.error.assert_called_once_with(
            "ERROR: Cannot download model data file"
        )
