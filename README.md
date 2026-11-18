# The Hedge Fund Lab

A Streamlit-based educational portfolio construction and risk-management simulator.

## Run locally

1. Install Python 3.10+.
2. In a terminal, move into this folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Launch:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL printed by Streamlit (normally http://localhost:8501).

## Features

- Student portfolio construction from a curated equity, ETF, and diversifier universe
- Historical adjusted-price download through Yahoo Finance (`yfinance`)
- Offline synthetic-data fallback clearly labeled as synthetic
- CAGR, annualized volatility, Sharpe, Sortino, downside deviation, maximum drawdown, VaR and CVaR
- Correlation matrix, return scatterplot, rolling correlation
- Historical market-regime windows and customizable sector shocks
- Maximum-Sharpe, minimum-variance, and inverse-volatility risk-parity allocation estimates
- Portfolio manager challenge rubric and investment committee memo
- CSV export of portfolio weights

## Important teaching notes

- This is an educational prototype, not investment advice or a production trading system.
- Synthetic fallback prices are illustrative and must never be presented as actual market history.
- The optimizer uses historical sample means/covariances and long-only constraints. These estimates are unstable and sensitive to the sample window.
- Historical regime windows are fixed date windows and only display when included in the selected data range.
- The simplified stress test does not model options, leverage financing, margin calls, liquidity, borrow costs, or nonlinear exposures.
- Before classroom deployment, instructors should add a consistent data vendor, data licensing review, saved student accounts/submissions, instructor dashboards, and more rigorous validation/backtesting.
