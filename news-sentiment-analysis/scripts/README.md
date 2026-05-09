# Scripts

| Script | Purpose |
|--------|---------|
| `data_loader.py` | Download stock price data via yfinance and load news CSV |

## Usage

```bash
# Download all default tickers (AAPL, AMZN, GOOG, META, MSFT, NVDA, TSLA)
python scripts/data_loader.py

# Or from Python:
from scripts.data_loader import download_stock_data, load_news_data
price_data = download_stock_data(tickers=['AAPL', 'TSLA'], start='2020-01-01')
news_df    = load_news_data('../data/raw/raw_analyst_ratings.csv')
```
