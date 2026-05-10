# Contributing Guide

Thank you for your interest in contributing to this project!

This project is designed for seismologists, geophysicists, students, and researchers who want to develop, test, and share AI models for seismology.

This guide explains the basic workflow for working with GitHub, writing code, running checks, and submitting your changes.

---

## 1. Before You Start

Before making a contribution, please make sure you have:

- A GitHub account
- Git installed on your computer
- Python installed
- Basic familiarity with the command line
- A local copy of this repository

---

## 2. Setting Up GitHub

### 2.1 Configure your Git identity

After installing Git, configure your name and email:

```bash
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
```

This information will appear in your commits.

### 2.2 Clone the repository

Clone the repository to your computer:

```bash
git clone git@github.com:pangruncheng/ASTF-Net.git
cd ASTF-Net
```

### 2.3 Keep your local repository up to date

Before starting new work, always update your local copy:

```bash
git checkout dev
git pull origin dev
```

If this repository uses `main` instead of `dev`, replace `dev` with `main`.

### 2.4 Install the development environment

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management. If `uv` is not installed yet, install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From the repository root, create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate
```

Install the package with development dependencies:

```bash
uv sync --extra dev
```

This installs `astfnet` in editable mode, so local source code changes take effect without reinstalling the package. The development dependencies include tools such as `pre-commit`, `pytest`, `pytest-cov`, `ruff`, and `ty`.

Install the pre-commit hooks:

```bash
pre-commit install
```

Run the test suite to confirm the environment is ready:

```bash
pytest tests/
```

---

## 3. Creating a Feature Branch

Please do not make changes directly on `dev` or `main`.

Instead, create a new **feature branch** for each task.

### 3.1 Create a branch

```bash
git checkout dev
git pull origin dev
git checkout -b feature/your-feature-name
```

Examples of good branch names:

```text
feature/add-phase-picker
feature/add-waveform-dataset-loader
fix/sampling-rate-bug
docs/update-installation-guide
test/add-transform-tests
```

Use short but meaningful names.

### 3.2 Make your changes

Edit the code, documentation, or tests in your branch.

You can check which files changed with:

```bash
git status
```

### 3.3 Save your changes with commits

Add your changed files:

```bash
git add path/to/file.py
```

Commit your changes:

```bash
git commit -m "Add waveform normalization utility"
```

Good commit messages should briefly explain what changed.

Good examples:

```text
Add dataset loader for STEAD waveforms
Fix incorrect sampling rate conversion
Add tests for phase picking metrics
Update installation instructions
```

Less helpful examples:

```text
update
fix bug
changes
final version
```

### 3.4 Push your branch to GitHub

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub.

---

## 4. Pull Request Workflow

A Pull Request, or PR, is a request to merge your changes into the main codebase.

When opening a PR, please include:

```md
## Summary

Briefly describe what this PR changes.

## Motivation

Why is this change useful?

## Changes

- Added ...
- Fixed ...
- Updated ...

## Testing

Describe how you tested the change.

## Related Issues (If Applicable)

Closes #123
```

Please keep PRs focused. A small PR that solves one clear problem is easier to review than a large PR that changes many unrelated things.

For large changes, please open an issue first to discuss the idea.

---

## 5. Pre-commit Checks

This project uses `pre-commit` to automatically check code quality before each commit. The hooks are configured in `.pre-commit-config.yaml`.

### 5.1 What is pre-commit?

`pre-commit` is a tool that runs automatic checks before your code is committed.

In this repository, pre-commit runs checks that:

- block accidentally committed large files
- detect private keys
- fix missing final newlines
- remove trailing whitespace
- run `ruff check --fix` on Python files
- run `ruff format` on Python files

Some hooks modify files automatically. If that happens, review the changes, add the modified files, and commit again.

### 5.2 Install pre-commit

If you installed the development environment with `uv sync --extra dev`, `pre-commit` and `ruff` are already available in your virtual environment. Otherwise, install `pre-commit` manually:

```bash
uv add pre-commit
```

Then install the hooks:

```bash
pre-commit install
```

After this, checks will run automatically whenever you commit.

### 5.3 Run pre-commit manually

You can also run all hooks manually:

```bash
pre-commit run --all-files
```

Because this project uses `ruff check --fix`, `ruff format`, `end-of-file-fixer`, and `trailing-whitespace`, pre-commit may update files when it runs. If pre-commit changes files, add those files again and commit:

```bash
git add .
git commit -m "Apply formatting fixes"
```

If a check fails, read the error message carefully. Common failures include large files, detected private keys, lint errors Ruff could not fix automatically, or formatting changes that need to be staged.

---

## 6. Code Style Guidelines

The goal is to make the code easy to read, easy to test, and easy for researchers to understand.

### 6.1 Use clear names

Use descriptive names instead of short or unclear names.

Good:

```python
sampling_rate_hz = 100
waveform_length = 3000
event_id = "abc123"
station_latitude = 45.5
```

Avoid:

```python
sr = 100
x = 3000
id = "abc123"
lat = 45.5
```

Short names like `x`, `y`, or `i` are acceptable only in very small local contexts.

### 6.2 Naming conventions

Use standard Python naming conventions:

```text
Functions:        snake_case
Variables:        snake_case
Classes:          PascalCase
Constants:        UPPER_CASE
Private helpers:  _leading_underscore
```

Examples:

```python
def load_waveform(...):
    ...

class WaveformDataset:
    ...

DEFAULT_SAMPLING_RATE_HZ = 100

def _normalize_trace(...):
    ...
```

### 6.3 Be explicit about units

Seismology code often involves physical quantities. Please include units in variable names when possible.

Good:

```python
sampling_rate_hz
distance_km
depth_km
time_window_sec
velocity_km_per_sec
```

Avoid:

```python
sampling_rate
distance
depth
window
velocity
```

### 6.4 Be explicit about array and tensor shapes

When working with NumPy arrays or PyTorch tensors, document the expected shape.

Example:

```python
# waveform shape: (num_channels, num_samples)
```

or in a docstring:

```python
Args:
    waveform: Input waveform with shape `(num_channels, num_samples)`.
```

For deep learning models, please clearly document whether the expected shape is:

```text
(batch, channels, time)
(batch, time, channels)
(num_samples,)
```

This avoids many common bugs.

### 6.5 Avoid hard-coded local paths

Do not write code that only works on your own computer.

Avoid:

```python
data_path = "/Users/myname/Desktop/seismic_data"
```

Prefer:

```python
data_path = Path(config.data_dir)
```

or allow the user to pass the path as an argument.

### 6.6 Keep functions focused

A function should usually do one clear thing.

Good:

```python
def normalize_waveform(waveform):
    ...
```

Avoid creating very large functions that load data, preprocess it, train a model, evaluate results, and save figures all at once.

---

## 7. Comments and Docstrings

This project uses **Google-style docstrings**.

A docstring is a block of text that explains what a function, class, or method does.

### 7.1 Function docstring format

Use this format:

```python
def normalize_waveform(waveform: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a waveform by removing the mean and scaling by standard deviation.

    Args:
        waveform: Input waveform with shape `(num_channels, num_samples)`.
        eps: Small value added to the standard deviation to avoid division by zero.

    Returns:
        Normalized waveform with the same shape as the input.
    """
    mean = waveform.mean(axis=-1, keepdims=True)
    std = waveform.std(axis=-1, keepdims=True)
    return (waveform - mean) / (std + eps)
```

### 7.2 Include important scientific assumptions

For seismology and AI code, please document assumptions such as:

- Expected sampling rate
- Input units
- Coordinate system
- Channel order
- Phase labels
- Whether data is filtered
- Whether normalization is applied per trace, per event, or per dataset

Example:

```python
def bandpass_filter_trace(
    trace: np.ndarray,
    sampling_rate_hz: float,
    freq_min_hz: float,
    freq_max_hz: float,
) -> np.ndarray:
    """Apply a bandpass filter to a single seismic trace.

    Args:
        trace: One-dimensional waveform array with shape `(num_samples,)`.
        sampling_rate_hz: Sampling rate in Hz.
        freq_min_hz: Lower corner frequency in Hz.
        freq_max_hz: Upper corner frequency in Hz.

    Returns:
        Filtered waveform with shape `(num_samples,)`.

    Notes:
        This function assumes the input trace has already been detrended.
    """
    ...
```

### 7.3 Comments should explain why, not only what

Good comments explain the reason behind the code.

Good:

```python
# Use per-trace normalization because station amplitudes may vary significantly.
waveform = normalize_waveform(waveform)
```

Less useful:

```python
# Normalize waveform.
waveform = normalize_waveform(waveform)
```

The code already says what is happening. The comment should explain why it matters.

---

## 8. Type Hints

Please use type hints when possible.

Example:

```python
def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    ...
```

Type hints help contributors understand what kind of input a function expects.

Common examples:

```python
from pathlib import Path
from typing import Optional

def load_metadata(path: Path) -> dict:
    ...

def get_event(event_id: str) -> Optional[dict]:
    ...
```

---

## 9. Testing Guidelines

Tests help make sure the code works and does not break when future changes are made.

This project uses `pytest`.

### 9.1 Run tests

To run all tests:

```bash
pytest
```

To run one test file:

```bash
pytest tests/test_waveform_utils.py
```

To run one specific test:

```bash
pytest tests/test_waveform_utils.py::test_normalize_waveform
```

### 9.2 Where to put tests

Test files should go in the `tests/` directory.

Use names like:

```text
tests/test_waveform_utils.py
tests/test_dataset_loader.py
tests/test_phase_picking_metrics.py
tests/test_model_forward.py
```

Test function names should start with `test_`.

Example:

```python
def test_normalize_waveform_has_zero_mean():
    ...
```

### 9.3 Write small tests

Each test should check one main behavior.

Good:

```python
def test_normalize_waveform_has_zero_mean():
    waveform = np.array([[1.0, 2.0, 3.0]])
    normalized = normalize_waveform(waveform)

    assert np.allclose(normalized.mean(axis=-1), 0.0)
```

Better to have several small tests than one very large test.

### 9.4 Use small synthetic data

Tests should be fast and easy to run.

Prefer small synthetic waveforms instead of large real datasets.

Good:

```python
waveform = np.random.randn(3, 100)
```

Avoid requiring large local files, private datasets, or long downloads in basic tests.

### 9.5 Test edge cases

Please test important edge cases, such as:

- Empty input
- Very short waveforms
- Missing metadata
- Different sampling rates
- NaN values
- CPU-only environment
- Single-channel versus three-channel input

### 9.6 Tests should be deterministic

Tests should produce the same result every time.

If using randomness, set a random seed:

```python
rng = np.random.default_rng(seed=42)
waveform = rng.normal(size=(3, 100))
```

For PyTorch:

```python
torch.manual_seed(42)
```

### 9.7 Testing machine learning code

For model code, tests do not need to prove that the model is scientifically good.

Basic tests should check that:

- The model can be initialized
- The forward pass works
- Input and output shapes are correct
- Loss computation works
- Training step does not crash on a tiny batch

Example:

```python
def test_model_forward_shape():
    model = MyPhasePickerModel()
    batch = torch.randn(2, 3, 3000)

    output = model(batch)

    assert output.shape[0] == 2
```

### 9.8 Avoid GPU requirements in basic tests

Basic tests should run on CPU unless there is a specific reason.

If a test requires GPU, mark it clearly.

```python
import pytest
import torch

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_model_on_gpu():
    ...
```

---

## 10. Documentation Updates

If your change affects how users interact with the project, please update the documentation.

Examples of changes that need documentation:

- New model
- New dataset loader
- New configuration option
- New command-line argument
- Changed input or output format
- Changed installation step

Documentation can be added to:

```text
README.md
docs/
examples/
notebooks/
```

---

## 11. Adding New Models

When adding a new AI model, please include:

- A clear model name
- A short description of the method
- Expected input shape
- Expected output shape
- Example usage
- Configuration options
- Basic tests
- References, if the model is based on a paper

Example docstring:

```python
class PhasePickerModel(nn.Module):
    """Neural network for seismic phase picking.

    Args:
        num_channels: Number of input waveform channels.
        hidden_dim: Hidden dimension of the model.
        num_classes: Number of output classes.

    Inputs:
        waveform: Tensor with shape `(batch, channels, time)`.

    Returns:
        Tensor with shape `(batch, time, num_classes)`.
    """
```

---

## 12. Adding Dataset Support

When adding support for a dataset, please document:

- Dataset name
- Data source
- License or usage restrictions
- Expected directory structure
- Required metadata fields
- Waveform format
- Sampling rate
- Channel order
- Label format
- Example loading code

Please do not commit large datasets directly to the repository.

For large files, use external storage, documented download scripts, or Git LFS if the project maintainers approve it.

---

## 13. What Not to Commit

Please avoid committing:

```text
Large raw datasets
Model checkpoints unless approved
Private data
API keys
Local environment files
Temporary files
Notebook outputs with huge embedded data
```

Common files that should not be committed:

```text
.env
.DS_Store
__pycache__/
*.pyc
.ipynb_checkpoints/
```

---

## 14. Getting Help

If you are unsure about anything, please open an issue or start a discussion.

Good questions include:

- "Where should I add this dataset loader?"
- "Is this model interface consistent with the rest of the project?"
- "Should this function be part of preprocessing or evaluation?"
- "What tests should I add for this change?"

We welcome contributions from people with different levels of software experience. Clear communication is more important than perfect code on the first try.

---

## 16. Checklist Before Opening a Pull Request

Before opening a PR, please check:

- [ ] My branch is based on the latest `dev` branch
- [ ] My code is focused on one clear change
- [ ] I used clear variable and function names
- [ ] I added or updated docstrings where needed
- [ ] I documented important assumptions, units, and shapes
- [ ] I added or updated tests
- [ ] I ran `pre-commit run --all-files`
- [ ] I ran `pytest`
- [ ] I updated documentation if needed
- [ ] I did not commit large data files or private files

Thank you for contributing!
