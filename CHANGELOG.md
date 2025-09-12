# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-09-12

### Fixed
- **Critical TensorFlow/Keras 3 compatibility issue**: Fixed model loading failures with TensorFlow 2.15+ and Keras 3
- Model SavedModel format incompatibility resolved by pinning TensorFlow to 2.10-2.14 range
- Deprecated `pkg_resources` imports replaced with `importlib.resources`

### Added
- `[tensorflow-compat]` optional dependency for users experiencing compatibility issues
- Comprehensive type hints throughout the codebase
- Enhanced error handling with descriptive error messages
- Logging system for better debugging and monitoring
- Complete test suite with 100+ tests covering all functionality
- GitHub Actions CI/CD pipeline with automated testing
- Support for Python 3.10, 3.11, and 3.12

### Changed
- **Breaking**: Minimum Python version raised to 3.10+
- Package status upgraded from Alpha to Beta
- Consolidated all dependencies in `pyproject.toml` (modern Python packaging)
- Improved documentation and installation instructions
- Enhanced CLI with better argument handling and error reporting

### Improved
- Code quality with Black formatting and MyPy type checking
- Test coverage and reliability
- Package structure and maintainability
- Documentation clarity and completeness

### Verified
- ✅ Successfully tested predictions on real names
- ✅ TensorFlow 2.14.1 + Keras 2.14.0 compatibility confirmed
- ✅ Model download and loading works correctly
- ✅ All core functionality operational

## [0.0.2] - Previous Release
- Initial release with basic functionality