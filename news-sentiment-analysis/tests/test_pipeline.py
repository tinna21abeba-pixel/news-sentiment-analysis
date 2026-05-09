"""
test_pipeline.py
================
Nova Financial Solutions – Unit Tests
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eda_utils import EDAAnalyzer
from src.sentiment_analysis import SentimentAnalyzer
from src.technical_indicators import TechnicalAnalyzer
from src.correlation_analysis import CorrelationAnalyzer


@pytest.fixture
def sample_news_df():
    """Minimal financial news dataset for testing."""
    return pd.DataFrame({
        "headline": [
            "Apple beats earnings expectations with record revenue",
            "Tesla misses Q3 delivery targets, shares fall",
            "NVIDIA reports strong AI chip demand outlook",
            "Amazon launches new cloud computing service",
            "Microsoft acquires gaming company for $10 billion",
            "",      
            None,    
        ],
        "url":       [f"https://example.com/{i}" for i in range(7)],
        "publisher": [
            "Reuters", "Benzinga", "Reuters",
            "analyst@benzinga.com", "MarketWatch",
            "Reuters", "Bloomberg"
        ],
        "date": pd.date_range("2023-01-02", periods=7, freq="D"),
        "stock": ["AAPL", "TSLA", "NVDA", "AMZN", "MSFT", "GOOG", "META"],
    })


@pytest.fixture
def sample_price_df():
    """Minimal OHLCV price data for testing."""
    np.random.seed(42)
    n = 300
    dates = pd.bdate_range("2022-01-03", periods=n)
    close = 150 + np.cumsum(np.random.randn(n) * 2)
    df = pd.DataFrame({
        "Date":      dates,
        "Open":      close * (1 + np.random.uniform(-0.005, 0.005, n)),
        "High":      close * (1 + np.random.uniform(0.001, 0.015, n)),
        "Low":       close * (1 - np.random.uniform(0.001, 0.015, n)),
        "Close":     close,
        "Adj Close": close * (1 - np.random.uniform(0, 0.002, n)),
        "Volume":    np.random.randint(1_000_000, 50_000_000, n).astype(float),
    })
    return df


class TestEDAAnalyzer:

    def test_init_drops_missing_rows(self, sample_news_df):
        """EDAAnalyzer should drop rows with null headline or date."""
        analyzer = EDAAnalyzer(sample_news_df)
        assert analyzer.df["headline"].notna().all()
        assert analyzer.df["date"].notna().all()

    def test_headline_len_column_created(self, sample_news_df):
        """headline_len column should be a positive integer."""
        analyzer = EDAAnalyzer(sample_news_df)
        assert "headline_len" in analyzer.df.columns
        assert (analyzer.df["headline_len"] > 0).all()

    def test_descriptive_stats_returns_dataframe(self, sample_news_df):
        analyzer = EDAAnalyzer(sample_news_df)
        stats = analyzer.descriptive_stats()
        assert isinstance(stats, pd.DataFrame)
        assert "headline_char_count" in stats.columns

    def test_publisher_analysis_top_n(self, sample_news_df):
        analyzer = EDAAnalyzer(sample_news_df)
        pub_df = analyzer.publisher_analysis(top_n=3)
        assert len(pub_df) <= 3
        assert "article_count" in pub_df.columns

    def test_email_domain_extraction(self, sample_news_df):
        analyzer = EDAAnalyzer(sample_news_df)
        pub_df = analyzer.publisher_analysis(top_n=10)
        email_row = pub_df[pub_df["publisher"] == "analyst@benzinga.com"]
        if len(email_row) > 0:
            assert email_row["domain"].values[0] == "benzinga.com"

    def test_news_volume_returns_sorted_daily(self, sample_news_df):
        analyzer = EDAAnalyzer(sample_news_df)
        daily = analyzer.news_volume_over_time()
        assert "article_count" in daily.columns
        dates = pd.to_datetime(daily["date_only"])
        assert dates.is_monotonic_increasing

    def test_tfidf_returns_top_n(self, sample_news_df):
        analyzer = EDAAnalyzer(sample_news_df)
        kw = analyzer.top_tfidf_keywords(top_n=5)
        assert len(kw) <= 5
        assert "keyword" in kw.columns
        assert "tfidf_score" in kw.columns



class TestSentimentAnalyzer:

    def test_score_headlines_adds_columns(self, sample_news_df):
        sa = SentimentAnalyzer(sample_news_df)
        df = sa.score_headlines()
        for col in ["vader_compound", "vader_pos", "vader_neg", "sentiment_label"]:
            assert col in df.columns

    def test_compound_score_range(self, sample_news_df):
        sa = SentimentAnalyzer(sample_news_df)
        df = sa.score_headlines()
        scores = df["vader_compound"].dropna()
        assert (scores >= -1).all() and (scores <= 1).all()

    def test_sentiment_labels_valid(self, sample_news_df):
        sa = SentimentAnalyzer(sample_news_df)
        df = sa.score_headlines()
        valid_labels = {"Positive", "Neutral", "Negative"}
        assert set(df["sentiment_label"].dropna().unique()).issubset(valid_labels)

    def test_positive_headline_is_positive(self):
        df = pd.DataFrame({
            "headline": ["Company beats earnings with record profit growth"],
            "url": ["http://x.com"], "publisher": ["Reuters"],
            "date": ["2023-01-03"], "stock": ["AAPL"]
        })
        sa = SentimentAnalyzer(df)
        scored = sa.score_headlines()
        assert scored["vader_compound"].values[0] > 0

    def test_negative_headline_is_negative(self):
        df = pd.DataFrame({
            "headline": ["Company misses revenue forecast badly, shares crash"],
            "url": ["http://x.com"], "publisher": ["Reuters"],
            "date": ["2023-01-03"], "stock": ["TSLA"]
        })
        sa = SentimentAnalyzer(df)
        scored = sa.score_headlines()
        assert scored["vader_compound"].values[0] < 0

    def test_date_alignment_no_weekend_trade_dates(self, sample_news_df):
        sa = SentimentAnalyzer(sample_news_df)
        sa.score_headlines()
        df_aligned = sa.align_to_trading_days()
        trade_dates = pd.to_datetime(df_aligned["trade_date"].dropna())
        assert (trade_dates.dt.dayofweek < 5).all()

    def test_aggregate_daily_sentiment_structure(self, sample_news_df):
        sa = SentimentAnalyzer(sample_news_df)
        sa.score_headlines()
        sa.align_to_trading_days()
        agg = sa.aggregate_daily_sentiment()
        for col in ["stock", "trade_date", "avg_sentiment", "article_count"]:
            assert col in agg.columns


class TestTechnicalAnalyzer:

    def test_prepare_sets_datetime_index(self, sample_price_df):
        ta = TechnicalAnalyzer(sample_price_df, ticker="TEST")
        assert isinstance(ta.df.index, pd.DatetimeIndex)

    def test_compute_all_indicators_columns(self, sample_price_df):
        ta = TechnicalAnalyzer(sample_price_df, ticker="TEST")
        df = ta.compute_all_indicators()
        for col in ["SMA_20", "SMA_50", "EMA_50", "RSI_14", "MACD",
                    "MACD_Signal", "MACD_Hist", "Daily_Return", "Cum_Return"]:
            assert col in df.columns

    def test_rsi_range(self, sample_price_df):
        ta = TechnicalAnalyzer(sample_price_df, ticker="TEST")
        ta.compute_all_indicators()
        rsi = ta.df["RSI_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_daily_return_is_percentage(self, sample_price_df):
        ta = TechnicalAnalyzer(sample_price_df, ticker="TEST")
        ta.compute_all_indicators()
        ret = ta.df["Daily_Return"].dropna()

        assert (ret.abs() < 30).mean() > 0.95

    def test_pynance_metrics_keys(self, sample_price_df):
        ta = TechnicalAnalyzer(sample_price_df, ticker="TEST")
        ta.compute_all_indicators()
        metrics = ta.compute_pynance_metrics()
        for key in ["Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Annualised Volatility"]:
            assert key in metrics


class TestCorrelationAnalyzer:

    @pytest.fixture
    def prepared_sentiment_df(self, sample_news_df):
        sa = SentimentAnalyzer(sample_news_df)
        sa.score_headlines()
        sa.align_to_trading_days()
    
        agg = sa.aggregate_daily_sentiment()
        agg["stock"] = "AAPL"
        return agg

    def test_merge_returns_dataframe(self, prepared_sentiment_df, sample_price_df):
        ca = CorrelationAnalyzer(prepared_sentiment_df, sample_price_df, ticker="AAPL")
        merged = ca.merge()
        assert isinstance(merged, pd.DataFrame)
        assert "avg_sentiment" in merged.columns
        assert "daily_return" in merged.columns

    def test_pearson_r_range(self, prepared_sentiment_df, sample_price_df):
        ca = CorrelationAnalyzer(prepared_sentiment_df, sample_price_df, ticker="AAPL")
        ca.merge()
        result = ca.pearson_correlation()
        assert -1 <= result["pearson_r"] <= 1

    def test_pearson_result_keys(self, prepared_sentiment_df, sample_price_df):
        ca = CorrelationAnalyzer(prepared_sentiment_df, sample_price_df, ticker="AAPL")
        ca.merge()
        result = ca.pearson_correlation()
        for key in ["ticker", "pearson_r", "p_value", "significant",
                    "n_observations", "ci_95_lower", "ci_95_upper"]:
            assert key in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
