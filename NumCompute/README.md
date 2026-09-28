## Project Setup

### 1. Install Conda

If you don't already have Conda installed, grab **Miniconda**:

- [Miniconda downloads](https://docs.conda.io/en/latest/miniconda.html)

Verify the install:

```bash
conda --version
```

### 2. Create and activate the virtual environment

From the project root:

```bash
conda create -n numComp_env python=3.12
conda activate numComp_env
```

### 3. Install the package (editable mode)

With the environment active, install `numcompute` and its dependencies:

```bash
pip install -e .
```

### 4. VSCode setup

1. Install the **Python**, **Pylance**, and **Jupyter** extentions.
2. Open the command palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) → **Python: Select Interpreter** → choose the `numComp_env` conda environment.
3. If IntelliSense/linting doesn't pick up the change right away, run **Developer: Reload Window** from the command palette.
4. Confirm the correct interpreter is active by opening a new integrated terminal (it should auto-activate `numComp_env`) and running:

   ```bash
   which python   # macOS/Linux
   where python   # Windows
   ```

   This should point into your conda env's directory, not the system Python.

### 5. Deactivating / removing the environment

```bash
conda deactivate
conda env remove -n numComp_env   # if you need to start fresh
```


## Running Tests

Tests are written using **pytest** and live under the `tests/` directory.


### 1. Run the full test suite

From the project root:

```bash
pytest
```

### 2. Run tests for a single module

```bash
pytest tests/test_io.py
```

### 3. Run a single test function

```bash
pytest tests/test_io.py::test_mean_basic
```

### 4. Verbose output

Shows each test name and pass/fail status individually:

```bash
pytest -v
```

### 5. Stop on first failure

Useful when debugging — stops immediately instead of running the whole suite:

```bash
pytest -x
```

### 6. Show print statements / output

By default pytest captures stdout; use `-s` to see it:

```bash
pytest -s
```

### 7. Check test coverage

```bash
pytest --cov=numcompute tests/
```

For a more detailed line-by-line report:

```bash
pytest --cov=numcompute --cov-report=term-missing tests/
```

### 8. Running tests in VSCode

1. Open the command palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) → **Python: Configure Tests** → select **pytest** → select the `tests/` directory.
2. Tests will then appear in the **Testing** sidebar (flask icon), where you can run/debug individual tests or the whole suite with one click.