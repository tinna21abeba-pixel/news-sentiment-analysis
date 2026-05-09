# Notebooks

This directory contains the three Jupyter notebooks for each task.

| Notebook | Task | Branch |
|----------|------|--------|
| `task1_eda.ipynb` | Exploratory Data Analysis | `task-1` |
| `task2_technical_indicators.ipynb` | Quantitative Analysis (TA-Lib + PyNance) | `task-2` |
| `task3_correlation_analysis.ipynb` | Sentiment & Correlation Analysis | `task-3` |

## Running Notebooks

```bash
cd news-sentiment-analysis
jupyter notebook notebooks/
```

Ensure you have run `scripts/data_loader.py` first to populate `data/raw/` with price CSVs.
