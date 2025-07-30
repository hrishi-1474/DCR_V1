# Contributing to Clean Room Data Processor

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Ollama with Phi-3.5 Mini model
- Basic knowledge of Streamlit and data processing

### Development Setup
```bash
# Clone the repository
git clone <repository-url>
cd DCR_PoC_V0

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Ollama service
brew services start ollama
ollama pull phi3
```

## 🔧 Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes
```bash
# Run all tests
python tests/run_tests.py

# Run specific tests
python tests/test_phi_functions.py
python tests/test_streamlit_app.py
```

### 4. Commit Your Changes
```bash
git add .
git commit -m "feat: add your feature description"
```

### 5. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

## 📋 Contribution Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings for new functions
- Keep functions focused and concise

### Testing
- Add tests for new functionality
- Ensure all existing tests pass
- Test with sample data files
- Test edge cases and error conditions

### Documentation
- Update README.md if adding new features
- Add comments for complex logic
- Update docstrings for modified functions

## 🐛 Reporting Issues

When reporting issues, please include:
- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, etc.)
- Error messages or logs

## 💡 Feature Requests

For feature requests, please:
- Describe the feature clearly
- Explain the use case
- Consider implementation complexity
- Check if similar features exist

## 📝 Pull Request Guidelines

### Before Submitting
- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] No sensitive data in commits

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Test addition

## Testing
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
```

## 🏷️ Commit Message Format

Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions
- `refactor:` for code refactoring

Example:
```
feat: add support for Parquet file format
fix: resolve memory leak in large file processing
docs: update installation instructions
```

## 🤝 Code Review Process

1. **Automated Checks**: CI/CD will run tests
2. **Review**: Maintainers will review your PR
3. **Feedback**: Address any feedback or requested changes
4. **Merge**: Once approved, your PR will be merged

## 📞 Getting Help

- **Issues**: Use GitHub Issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check README.md and QUICK_START.md

## 🙏 Thank You

Thank you for contributing to the Clean Room Data Processor! Your contributions help make this tool better for everyone. 