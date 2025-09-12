"""Tests for base module."""

import pytest
import os
import tempfile
from unittest.mock import patch, Mock

from pranaam.base import Base


class TestBase:
    """Test Base class functionality."""
    
    def test_base_class_attributes(self):
        """Test Base class has expected attributes."""
        assert hasattr(Base, 'MODELFN')
        assert hasattr(Base, 'load_model_data')
        assert Base.MODELFN is None
    
    def test_load_model_data_no_modelfn(self):
        """Test load_model_data when MODELFN is None."""
        # Create a test class without MODELFN set
        class TestClass(Base):
            MODELFN = None
        
        result = TestClass.load_model_data("test_file")
        assert result is None
    
    @patch('pranaam.base.files')
    @patch('pranaam.base.download_file')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_load_model_data_success(self, mock_makedirs, mock_exists, mock_download, mock_files):
        """Test successful model data loading."""
        # Setup mocks
        mock_files.return_value.__truediv__.return_value = "/fake/model/path"
        
        mock_exists.side_effect = lambda path: path == "/fake/model/path"  # Dir exists, file doesn't
        mock_download.return_value = True
        
        # Create test class
        class TestClass(Base):
            MODELFN = "model"
        
        result = TestClass.load_model_data("test_model", latest=False)
        
        assert result == "/fake/model/path"
        mock_makedirs.assert_called_once_with("/fake/model/path")
        mock_download.assert_called_once()
    
    @patch('pranaam.base.files')
    @patch('pranaam.base.download_file')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_load_model_data_file_exists_no_latest(self, mock_makedirs, mock_exists, mock_download, mock_files):
        """Test model loading when file exists and latest=False."""
        mock_files.return_value.__truediv__.return_value = "/fake/model/path"
        
        # Both directory and file exist
        mock_exists.return_value = True
        
        class TestClass(Base):
            MODELFN = "model"
        
        result = TestClass.load_model_data("test_model", latest=False)
        
        assert result == "/fake/model/path"
        # Should not download since file exists and latest=False
        mock_download.assert_not_called()
    
    @patch('pranaam.base.files')
    @patch('pranaam.base.download_file')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_load_model_data_force_latest(self, mock_makedirs, mock_exists, mock_download, mock_files):
        """Test model loading with latest=True forces redownload."""
        mock_files.return_value.__truediv__.return_value = "/fake/model/path"
        
        mock_exists.return_value = True  # File exists
        mock_download.return_value = True
        
        class TestClass(Base):
            MODELFN = "model"
        
        result = TestClass.load_model_data("test_model", latest=True)
        
        assert result == "/fake/model/path"
        # Should download even though file exists because latest=True
        mock_download.assert_called_once()
    
    @patch('pranaam.base.files')
    @patch('pranaam.base.download_file')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_load_model_data_download_failure(self, mock_makedirs, mock_exists, mock_download, mock_files):
        """Test handling of download failure."""
        mock_files.return_value.__truediv__.return_value = "/fake/model/path"
        
        mock_exists.side_effect = lambda path: path == "/fake/model/path"  # Dir exists, file doesn't
        mock_download.return_value = False  # Download fails
        
        class TestClass(Base):
            MODELFN = "model"
        
        result = TestClass.load_model_data("test_model")
        
        # Should still return path even if download fails
        assert result == "/fake/model/path"
    
    @patch('pranaam.base.files')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_load_model_data_creates_directory(self, mock_makedirs, mock_exists, mock_files):
        """Test that model directory is created if it doesn't exist."""
        mock_files.return_value.__truediv__.return_value = "/fake/model/path"
        
        mock_exists.return_value = False  # Directory doesn't exist
        
        class TestClass(Base):
            MODELFN = "model"
        
        with patch('pranaam.base.download_file', return_value=True):
            result = TestClass.load_model_data("test_model")
        
        mock_makedirs.assert_called_once_with("/fake/model/path")
        assert result == "/fake/model/path"


class TestBaseInheritance:
    """Test Base class inheritance patterns."""
    
    def test_subclass_can_override_modelfn(self):
        """Test that subclasses can override MODELFN."""
        class CustomBase(Base):
            MODELFN = "custom_model"
        
        assert CustomBase.MODELFN == "custom_model"
        assert Base.MODELFN is None  # Original unchanged
    
    def test_multiple_subclasses_independent(self):
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
    
    @patch('pranaam.base.logger')
    @patch('pranaam.base.files')
    @patch('pranaam.base.download_file')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_debug_logging_download(self, mock_makedirs, mock_exists, mock_download, mock_files, mock_logger):
        """Test debug logging during download."""
        mock_files.return_value.__truediv__.return_value = "/fake/model/path"
        
        mock_exists.side_effect = lambda path: path == "/fake/model/path"  # Dir exists, file doesn't
        mock_download.return_value = True
        
        class TestClass(Base):
            MODELFN = "model"
        
        TestClass.load_model_data("test_model")
        
        # Should log download message
        mock_logger.debug.assert_called()
        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Downloading model data" in call for call in debug_calls)
    
    @patch('pranaam.base.logger')
    @patch('pranaam.base.files')
    @patch('os.path.exists')
    def test_debug_logging_existing_model(self, mock_exists, mock_files, mock_logger):
        """Test debug logging when using existing model."""
        mock_files.return_value.__truediv__.return_value = "/fake/model/path"
        
        mock_exists.return_value = True  # File exists
        
        class TestClass(Base):
            MODELFN = "model"
        
        TestClass.load_model_data("test_model")
        
        # Should log using existing model message
        mock_logger.debug.assert_called()
        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Using model data from" in call for call in debug_calls)
    
    @patch('pranaam.base.logger')
    @patch('pranaam.base.files')
    @patch('pranaam.base.download_file')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_error_logging_download_failure(self, mock_makedirs, mock_exists, mock_download, mock_files, mock_logger):
        """Test error logging when download fails."""
        mock_files.return_value.__truediv__.return_value = "/fake/model/path"
        
        mock_exists.side_effect = lambda path: path == "/fake/model/path"  # Dir exists, file doesn't
        mock_download.return_value = False  # Download fails
        
        class TestClass(Base):
            MODELFN = "model"
        
        TestClass.load_model_data("test_model")
        
        # Should log error message
        mock_logger.error.assert_called_once_with("ERROR: Cannot download model data file")