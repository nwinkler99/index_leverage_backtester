# Investment Dashboard (Notebook)

This notebook downloads MSCI World, Gold, and VIX data from yfinance and builds an interactive backtest dashboard with custom leverage formulas.

## Setup

```bash
pip install -r requirements.txt
```

## Run

Open [dashboard.ipynb](dashboard.ipynb) in VS Code and run all cells.

## Notes

- Default tickers: MSCI World = `URTH`, Gold = `GLD`, VIX = `^VIX`.
- Use the formula box to define custom leverage rules. Supported functions: `min`, `max`, `abs`, `where`.
- Exports are saved to the `exports/` folder (CSV + HTML; PNG if `kaleido` is installed).
