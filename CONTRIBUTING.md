# Contributing to vtpl

Thank you for your interest in improving **vtpl**!

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/<your-username>/video-template-extractor.git
   cd video-template-extractor
   ```

2. Create and activate a Python 3.11+ virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
   ```

3. Install in editable mode with development and web dependencies:
   ```bash
   pip install -e '.[dev,web]'
   ```

4. Run the test suite:
   ```bash
   pytest -m "not slow"
   ```

## Pull Request Guidelines

1. Create a descriptive feature branch: `git checkout -b feature/my-enhancement`.
2. Ensure all unit tests pass before opening a PR.
3. Keep pull requests focused on a single issue or feature.
