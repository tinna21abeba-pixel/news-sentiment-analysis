"""
sentiment_analysis.py
=====================
Nova Financial Solutions – Task 3 (Part A)
Apply NLTK VADER to assign sentiment scores to financial headlines.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import warnings

warnings.filterwarnings("ignore")

# Download VADER lexicon if not present
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = sns.color_palette("muted")


class SentimentAnalyzer:
    """
    Assigns VADER sentiment scores to financial news headlines
    and classifies them as Positive, Neutral, or Negative.

    Why VADER?
    ----------
    VADER (Valence Aware Dictionary and sEntiment Reasoner) is specifically
    tuned for short social-media / news texts and understands financial
    qualifiers (e.g., "beats expectations", "misses forecast") better than
    general-purpose models. It requires no training data, making it ideal
    for rapid deployment.

    Parameters
    ----------
    df : pd.DataFrame
        News dataframe with at minimum columns: headline, date, stock
    """

    POSITIVE_THRESHOLD = 0.05
    NEGATIVE_THRESHOLD = -0.05

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._sia = SentimentIntensityAnalyzer()

    # ── Core Scoring ─────────────────────────────────────────────────────────
    def score_headlines(self) -> pd.DataFrame:
        """
        Apply VADER to each headline and return the enriched DataFrame.
        New columns: vader_neg, vader_neu, vader_pos, vader_compound, sentiment_label
        """
        scores = self.df["headline"].dropna().apply(
            lambda h: self._sia.polarity_scores(str(h))
        )
        scores_df = pd.DataFrame(scores.tolist(), index=scores.index)
        self.df = self.df.join(
            scores_df.rename(columns={
                "neg": "vader_neg",
                "neu": "vader_neu",
                "pos": "vader_pos",
                "compound": "vader_compound",
            }),
            how="left"
        )
        self.df["vader_compound"].fillna(0, inplace=True)
        self.df["sentiment_label"] = self.df["vader_compound"].apply(self._classify)

        pos = (self.df["sentiment_label"] == "Positive").sum()
        neu = (self.df["sentiment_label"] == "Neutral").sum()
        neg = (self.df["sentiment_label"] == "Negative").sum()
        print(f"[Sentiment] Positive: {pos:,} | Neutral: {neu:,} | Negative: {neg:,}")
        return self.df

    def _classify(self, compound: float) -> str:
        if compound >= self.POSITIVE_THRESHOLD:
            return "Positive"
        elif compound <= self.NEGATIVE_THRESHOLD:
            return "Negative"
        return "Neutral"

    # ── Normalise Dates to Trading Days ──────────────────────────────────────
    def align_to_trading_days(self, trading_calendar: pd.DatetimeIndex = None) -> pd.DataFrame:
        """
        Normalise publication dates:
          - Strip timezone info → date-only
          - Weekend / holiday articles → forwarded to next trading Monday

        Parameters
        ----------
        trading_calendar : DatetimeIndex, optional
            Pass the actual trading days from yfinance data for precision.
            If None, weekends are rolled forward to Monday.
        """
        self.df["date"] = pd.to_datetime(self.df["date"], utc=True, errors="coerce")
        self.df["trade_date"] = self.df["date"].dt.normalize().dt.tz_localize(None)

        if trading_calendar is not None:
            trading_set = set(trading_calendar.normalize())

            def snap_to_trading_day(d):
                if pd.isnull(d):
                    return pd.NaT
                for i in range(7):
                    candidate = d + pd.Timedelta(days=i)
                    if candidate in trading_set:
                        return candidate
                return pd.NaT

            self.df["trade_date"] = self.df["trade_date"].apply(snap_to_trading_day)
        else:
            # Simple weekend roll-forward
            dow = self.df["trade_date"].dt.dayofweek  # Mon=0, Sun=6
            self.df["trade_date"] = self.df.apply(
                lambda r: r["trade_date"] + pd.Timedelta(days=(7 - r["trade_date"].dayofweek) % 7)
                if r["trade_date"].dayofweek >= 5 else r["trade_date"],
                axis=1
            )

        self.df.dropna(subset=["trade_date"], inplace=True)
        print(f"[Date Align] Dates normalised. Shape: {self.df.shape}")
        return self.df

    # ── Daily Aggregation ─────────────────────────────────────────────────────
    def aggregate_daily_sentiment(self) -> pd.DataFrame:
        """
        Aggregate sentiment per (stock, trade_date):
          - mean compound score
          - article count
          - dominant sentiment label
        """
        agg = (
            self.df.groupby(["stock", "trade_date"])
            .agg(
                avg_sentiment=("vader_compound", "mean"),
                article_count=("headline", "count"),
                std_sentiment=("vader_compound", "std"),
            )
            .reset_index()
        )
        agg["sentiment_label"] = agg["avg_sentiment"].apply(self._classify)
        return agg

    # ── Visualisations ────────────────────────────────────────────────────────
    def plot_sentiment_distribution(self, save_path: str = None):
        """Bar chart + histogram of VADER compound scores."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Label counts
        label_counts = self.df["sentiment_label"].value_counts()
        colors = {"Positive": "#2ca02c", "Neutral": "#7f7f7f", "Negative": "#d62728"}
        bar_colors = [colors.get(l, "steelblue") for l in label_counts.index]
        axes[0].bar(label_counts.index, label_counts.values, color=bar_colors, edgecolor="white")
        for i, v in enumerate(label_counts.values):
            axes[0].text(i, v + 50, f"{v:,}", ha="center", fontsize=10)
        axes[0].set_title("Headline Sentiment Distribution", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Count")

        # Compound score distribution
        axes[1].hist(self.df["vader_compound"].dropna(), bins=80,
                     color=PALETTE[0], edgecolor="white", alpha=0.85)
        axes[1].axvline(0.05, color="#2ca02c", linestyle="--", linewidth=1.5,
                        label="Positive threshold (+0.05)")
        axes[1].axvline(-0.05, color="#d62728", linestyle="--", linewidth=1.5,
                        label="Negative threshold (-0.05)")
        axes[1].set_title("VADER Compound Score Distribution", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Compound Score")
        axes[1].set_ylabel("Frequency")
        axes[1].legend(fontsize=9)

        plt.suptitle("Nova Financial Solutions – News Sentiment Overview",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_sentiment_by_stock(self, top_n: int = 10, save_path: str = None):
        """Average VADER compound score per stock ticker."""
        stock_sent = (
            self.df.groupby("stock")["vader_compound"]
            .mean()
            .sort_values(ascending=False)
        )
        top    = stock_sent.head(top_n)
        bottom = stock_sent.tail(top_n)
        combined = pd.concat([top, bottom])

        fig, ax = plt.subplots(figsize=(12, 7))
        bar_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in combined.values]
        ax.barh(combined.index, combined.values, color=bar_colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Average VADER Compound Score")
        ax.set_title(
            f"Most Positive vs Most Negative Stocks by Avg. Sentiment",
            fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
