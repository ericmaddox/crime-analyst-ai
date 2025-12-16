# Contributing to Crime Analyst AI

First off, thank you for considering contributing to Crime Analyst AI! It's people like you that make this tool better for everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Style Guidelines](#style-guidelines)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to eric.maddox@outlook.com.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment (see below)
4. Create a branch for your changes
5. Make your changes
6. Push to your fork and submit a pull request

## How Can I Contribute?

### Reporting Bugs

- Use the GitHub issue tracker
- Use the bug report template
- Include as much detail as possible
- Include steps to reproduce the issue

### Suggesting Features

- Use the GitHub issue tracker
- Use the feature request template
- Explain the use case clearly
- Consider if this aligns with the project's goals

### Code Contributions

- Look for issues labeled `good first issue` or `help wanted`
- Comment on the issue to let others know you're working on it
- Follow the development setup and style guidelines

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ericmaddox/crime-analyst-ai.git
   cd crime-analyst-ai
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama and pull the model:**
   ```bash
   ollama pull ministral-3:3b
   ```

5. **Run the application:**
   ```bash
   python run.py
   ```

## Style Guidelines

### Python Code

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions focused and under 50 lines when possible
- Use meaningful variable and function names

### Example:

```python
def compute_crime_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive statistics about the crime data.
    
    Args:
        df: DataFrame with Latitude, Longitude, CrimeType columns
        
    Returns:
        Dictionary containing crime statistics
    """
    # Implementation here
```

### HTML/CSS (Gradio UI)

- Use CSS custom properties for theming
- Keep CSS organized with comments
- Follow the existing dark theme aesthetic

## Commit Messages

Use clear, descriptive commit messages:

- **feat:** A new feature
- **fix:** A bug fix
- **docs:** Documentation changes
- **style:** Code style changes (formatting, etc.)
- **refactor:** Code refactoring
- **test:** Adding or updating tests
- **chore:** Maintenance tasks

Example:
```
feat: Add temporal pattern analysis for crime predictions

- Analyze day of week patterns
- Analyze time of day patterns
- Include temporal context in LLM prompt
```

## Pull Request Process

1. **Before submitting:**
   - Ensure your code follows the style guidelines
   - Test your changes locally
   - Update documentation if needed

2. **Submitting:**
   - Fill out the PR template completely
   - Link any related issues
   - Request review from maintainers

3. **After submitting:**
   - Respond to review feedback
   - Make requested changes
   - Keep the PR up to date with main branch

4. **Merge:**
   - PRs require at least one approval
   - All CI checks must pass
   - Squash and merge is preferred

## Questions?

Feel free to reach out by opening an issue or contacting the maintainers directly.

Thank you for contributing! 🎉

