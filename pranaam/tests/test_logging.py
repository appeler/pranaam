"""Tests for logging module."""

import pytest
import logging
from unittest.mock import patch

from pranaam.logging import get_logger


class TestGetLogger:
    """Test get_logger function."""
    
    def test_default_logger_name(self):
        """Test getting logger with default name."""
        logger = get_logger()
        assert logger.name == "pranaam"
        assert isinstance(logger, logging.Logger)
    
    def test_custom_logger_name(self):
        """Test getting logger with custom name."""
        logger = get_logger("custom_name")
        assert logger.name == "custom_name"
        assert isinstance(logger, logging.Logger)
    
    def test_logger_configuration(self):
        """Test that logger is properly configured."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = mock_get_logger.return_value
            mock_logger.handlers = []  # No existing handlers
            
            get_logger("test")
            
            # Should add handler and set level
            mock_logger.addHandler.assert_called_once()
            mock_logger.setLevel.assert_called_once_with(logging.INFO)
    
    def test_no_duplicate_handlers(self):
        """Test that handlers are not duplicated on multiple calls."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = mock_get_logger.return_value
            mock_logger.handlers = [logging.StreamHandler()]  # Already has handler
            
            get_logger("test")
            
            # Should not add another handler
            mock_logger.addHandler.assert_not_called()
            mock_logger.setLevel.assert_not_called()
    
    def test_handler_formatter(self):
        """Test that handler has proper formatter."""
        # Clear any existing loggers to ensure clean test
        logger_name = "test_formatter_logger"
        if logger_name in logging.Logger.manager.loggerDict:
            del logging.Logger.manager.loggerDict[logger_name]
        
        logger = get_logger(logger_name)
        
        # Should have at least one handler
        assert len(logger.handlers) >= 1
        
        # Get the first handler (should be our StreamHandler)
        handler = logger.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        
        # Check formatter
        formatter = handler.formatter
        assert formatter is not None
        
        # Test formatter format string
        format_str = formatter._fmt
        expected_components = ["%(asctime)s", "%(name)s", "%(levelname)s", "%(message)s"]
        for component in expected_components:
            assert component in format_str
    
    def test_logger_level(self):
        """Test that logger is set to INFO level."""
        logger_name = "test_level_logger" 
        if logger_name in logging.Logger.manager.loggerDict:
            del logging.Logger.manager.loggerDict[logger_name]
            
        logger = get_logger(logger_name)
        assert logger.level == logging.INFO
    
    def test_same_logger_instance(self):
        """Test that same logger name returns same instance."""
        logger1 = get_logger("same_name")
        logger2 = get_logger("same_name")
        
        assert logger1 is logger2
    
    def test_different_logger_instances(self):
        """Test that different logger names return different instances.""" 
        logger1 = get_logger("name1")
        logger2 = get_logger("name2")
        
        assert logger1 is not logger2
        assert logger1.name != logger2.name


class TestLoggerFunctionality:
    """Test actual logging functionality."""
    
    def test_logger_can_log_messages(self, caplog):
        """Test that logger can actually log messages."""
        logger = get_logger("test_logging")
        
        with caplog.at_level(logging.INFO):
            logger.info("Test info message")
            logger.warning("Test warning message") 
            logger.error("Test error message")
        
        # Check that messages were logged
        assert "Test info message" in caplog.text
        assert "Test warning message" in caplog.text
        assert "Test error message" in caplog.text
    
    def test_logger_debug_level_filtering(self, caplog):
        """Test that DEBUG messages are filtered out by default."""
        logger = get_logger("test_debug")
        
        # Test at INFO level (default) - debug should be filtered
        with caplog.at_level(logging.INFO):
            logger.debug("Debug message")
            logger.info("Info message")
        
        # Debug should NOT appear at INFO level
        assert "Debug message" not in caplog.text
        assert "Info message" in caplog.text
        
        # Clear and test at DEBUG level - debug should appear
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            logger.debug("Debug message")
            logger.info("Info message")
        
        # Both should appear at DEBUG level
        assert "Debug message" in caplog.text
        assert "Info message" in caplog.text
    
    def test_logger_formatting_output(self, caplog):
        """Test that log messages are properly formatted."""
        logger = get_logger("test_format")
        
        with caplog.at_level(logging.INFO):
            logger.info("Test message")
        
        # Check format contains expected components
        log_output = caplog.text
        assert "test_format" in log_output  # Logger name
        assert "INFO" in log_output         # Log level
        assert "Test message" in log_output # Actual message
        # Note: timestamp format may vary, so we don't test exact format


class TestLoggerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_logger_with_empty_name(self):
        """Test logger with empty string name."""
        logger = get_logger("")
        # Empty string gets converted to "pranaam" by our function
        assert logger.name == "pranaam"  
        assert isinstance(logger, logging.Logger)
    
    def test_logger_with_none_name(self):
        """Test logger with None name (should use default)."""
        logger = get_logger(None)
        assert logger.name == "pranaam"
    
    def test_logger_with_special_characters(self):
        """Test logger name with special characters."""
        special_name = "test.logger-with_special123"
        logger = get_logger(special_name)
        assert logger.name == special_name
    
    @patch('logging.StreamHandler')
    def test_handler_creation_error(self, mock_handler):
        """Test handling of handler creation errors."""
        mock_handler.side_effect = Exception("Handler creation failed")
        
        # Should raise exception since we don't handle this error
        with pytest.raises(Exception, match="Handler creation failed"):
            get_logger("test_handler_error")
        
    def test_multiple_calls_same_name(self):
        """Test multiple calls with same name don't create duplicate config."""
        logger_name = "test_multiple_calls"
        
        # Clear existing logger
        if logger_name in logging.Logger.manager.loggerDict:
            del logging.Logger.manager.loggerDict[logger_name]
        
        logger1 = get_logger(logger_name)
        initial_handlers = len(logger1.handlers)
        
        logger2 = get_logger(logger_name)
        final_handlers = len(logger2.handlers)
        
        # Should not add more handlers
        assert initial_handlers == final_handlers
        assert logger1 is logger2