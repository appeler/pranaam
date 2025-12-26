# Contributing

We welcome contributions to pranaam! This guide will help you get started.

## Code of Conduct

The project welcomes contributions from everyone! It depends on it. To maintain this welcoming atmosphere and to collaborate in a fun and productive way, we expect contributors to the project to abide by the [Contributor Code of Conduct](http://contributor-covenant.org/version/1/0/0/).

## Getting Started

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:

   ```bash
   git clone https://github.com/yourusername/pranaam.git
   cd pranaam
   ```

3. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:

   ```bash
   pip install -e .[dev,test,docs]
   ```

5. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

## Development Workflow

### Code Quality Standards

We maintain high code quality standards:

* **Type Safety**: 100% MyPy compliance required
* **Code Formatting**: Black formatting (line length 88)
* **Testing**: Comprehensive test coverage
* **Documentation**: All public APIs must be documented

### Running Tests

Run the full test suite:

```bash
pytest
```

Run specific test categories:

```bash
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests
```

Run with coverage:

```bash
pytest --cov=pranaam --cov-report=html
```

### Code Quality Checks

Format code with Black:

```bash
black pranaam/
```

Type check with MyPy:

```bash
mypy pranaam/
```

Both commands must pass without errors before submitting a PR.

## Types of Contributions

### Bug Reports

When reporting bugs, please include:

* Python version and operating system
* TensorFlow version
* Complete error traceback
* Minimal code example to reproduce the issue
* Expected vs. actual behavior

### Feature Requests

Before submitting feature requests:

* Check existing issues and discussions
* Provide clear use case and rationale
* Consider implementation complexity
* Discuss API design implications

### Code Contributions

Areas where we welcome contributions:

* **New Language Support**: Adding support for additional Indian languages
* **Model Improvements**: Better accuracy or efficiency
* **Performance Optimizations**: Faster prediction times
* **Documentation**: Improved examples and guides
* **Testing**: Additional test cases and edge cases
* **Bug Fixes**: Resolving reported issues

## Submission Guidelines

### Pull Request Process

1. Create a feature branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our coding standards
3. Add or update tests as needed
4. Update documentation if applicable
5. Ensure all tests pass:

   ```bash
   pytest
   black --check pranaam/
   mypy pranaam/
   ```

6. Commit your changes with clear commit messages
7. Push to your fork and submit a pull request

### Pull Request Requirements

Your PR must:

* Pass all CI checks (tests, linting, type checking)
* Include appropriate tests for new functionality
* Update documentation for API changes
* Follow semantic versioning principles
* Include a clear description of changes

## Code Style Guidelines

### Python Code Style

* Follow PEP 8 with Black formatting
* Use type hints for all function signatures
* Write docstrings for all public functions and classes
* Maximum line length: 88 characters
* Use meaningful variable and function names

### Documentation Style

* Use Markdown (.md) format
* Include code examples for new features
* Write clear, concise explanations
* Update API documentation for code changes

### Testing Guidelines

* Write unit tests for all new functions
* Include integration tests for complex features
* Test edge cases and error conditions
* Mock external dependencies appropriately
* Aim for high test coverage (>90%)

## Project Structure

### Understanding the Codebase

```text
pranaam/
├── __init__.py              # Package initialization
├── naam.py                  # Core Naam class with pred_rel method
├── base.py                  # Base class for model data management
├── utils.py                 # Utility functions
├── logging.py               # Centralized logging configuration
├── pranaam.py              # CLI entry point and function exports
└── tests/                  # Comprehensive test suite
    ├── conftest.py         # pytest fixtures
    ├── test_naam.py        # Core functionality tests
    ├── test_integration.py # End-to-end integration tests
    └── ...                 # Additional test modules
```

### Key Components

* **naam.py**: Core prediction logic and model loading
* **base.py**: Model data management using importlib.resources
* **utils.py**: Helper functions for data processing
* **logging.py**: Centralized logging configuration
* **tests/**: Comprehensive test suite with 75+ tests

## Release Process

### Version Management

We follow semantic versioning (MAJOR.MINOR.PATCH):

* **MAJOR**: Breaking API changes
* **MINOR**: New features, backward compatible
* **PATCH**: Bug fixes, backward compatible

### Release Checklist

Before releasing a new version:

1. Update version in `pyproject.toml`
2. Update `CLAUDE.md` with changes and test status
3. Run full test suite: `pytest` (must be 75/75 passing)
4. Check formatting: `black --check pranaam/`
5. Type check: `mypy pranaam/` (must pass with zero errors)
6. Build package: `python -m build`
7. Validate: `python -m twine check dist/*`
8. Test in clean environment
9. Verify CI passes on GitHub Actions

## Communication

### Getting Help

* **GitHub Issues**: Bug reports and feature requests
* **GitHub Discussions**: General questions and ideas
* **Documentation**: Check our comprehensive docs first

### Maintainer Response

We aim to:

* Acknowledge issues within 48 hours
* Review pull requests within 1 week
* Provide constructive feedback
* Maintain respectful, professional communication

## Recognition

Contributors are recognized in:

* Release notes for significant contributions
* `AUTHORS.md` file (if we create one)
* GitHub contributors page

Thank you for contributing to pranaam! Your efforts help make this tool better for researchers, activists, and developers working to understand and address bias in AI systems.
