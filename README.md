# yfinance_api3

Reusable quantitative finance library built around `yfinance`.

## Project Layout

- `yfinance_api3/`: importable library code
- `notebooks/`: exploratory notebooks and generated analysis artifacts
- notebooks are kept outside the package so they do not become part of the distributed API surface

## Install

For local development:

```powershell
pip install -e .
```

For dashboard development:

```powershell
pip install -e .[dashboard-dash]
pip install -e .[dashboard-streamlit]
```

For linting, tests, and packaging tools:

```powershell
pip install -e .[dev]
```

Build a distributable wheel:

```powershell
python -m build
```

The built artifacts are created in `dist/`.

Run the Dash app:

```powershell
python -m yfinance_api3.dashboard.app
```

Run the Streamlit app:

```powershell
streamlit run streamlit_app.py
```

## Use In Another Project

Install from a local path:

```powershell
pip install C:\path\to\yfinanceAPI3
```

Or from Git:

```powershell
pip install git+https://github.com/<your-user>/<your-repo>.git
```

Import it in Python:

```python
from yfinance_api3 import QuantAnalytics, StockClient

client = StockClient()
qa = QuantAnalytics(client)
```


