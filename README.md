# crypto-strategies

Internal repository for systematic crypto trading strategies and backtests.

## Setup
```bash
git clone https://github.com/your-org/crypto-strategies.git
cd crypto-strategies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data
Data files are not committed to this repo.
See `data/README.md` for instructions on obtaining and placing data files.

## Adding a Strategy
1. Branch off `dev`: `git checkout -b feat/your-strategy-name`
2. Inherit from `BaseStrategy` in `strategies/base.py`
3. Add tests in `tests/`
4. Run backtest and save summary to `research/reports/`
5. Open PR to `dev`

## Structure
```
strategies/     # strategy logic only, no backtest code
backtesting/    # engine, metrics, reporting
research/       # notebooks and written reports
tests/          # unit tests
data/           # local only, gitignored
results/        # local only, gitignored
```
