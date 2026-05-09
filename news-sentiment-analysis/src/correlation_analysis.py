
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = sns.color_palette("muted")
COLORS = {"Positive": "#2ca02c", "Neutral": "#7f7f7f", "Negative": "#d62728"}


class CorrelationAnalyzer:
    """
    Merges daily sentiment scores with daily stock returns and
    measures their Pearson correlation.

    Parameters
    ----------
    sentiment_df : pd.DataFrame
        Output of SentimentAnalyzer.aggregate_daily_sentiment()
        Columns: stock, trade_date, avg_sentiment, article_count, sentiment_label
    price_df : pd.DataFrame
        Raw price DataFrame with columns: Date (index), Adj Close, (and ticker col or pass ticker)
    ticker : str
        Stock ticker to filter from price_df.
    """

    def __init__(self, sentiment_df: pd.DataFrame, price_df: pd.DataFrame, ticker: str):
        self.ticker = ticker
        self.sentiment_df = sentiment_df[sentiment_df["stock"] == ticker].copy()
        self.price_df     = price_df.copy()
        self.merged        = None
        self._compute_returns()
        
    def _compute_returns(self):
        """Calculate daily percentage return from Adj Close."""
        self.price_df["Date"] = pd.to_datetime(self.price_df.index
                                               if self.price_df.index.dtype != "object"
                                               else self.price_df["Date"])
        if not isinstance(self.price_df.index, pd.DatetimeIndex):
            self.price_df["Date"] = pd.to_datetime(self.price_df["Date"])
            self.price_df.set_index("Date", inplace=True)

        self.price_df.sort_index(inplace=True)
        self.price_df["daily_return"] = self.price_df["Adj Close"].pct_change() * 100
        self.price_df["trade_date"]   = self.price_df.index.normalize()
        print(f"[{self.ticker}] Returns computed over {len(self.price_df):,} trading days.")

   
    def merge(self) -> pd.DataFrame:
        """Inner-join sentiment with returns on trade_date."""
        self.sentiment_df["trade_date"] = pd.to_datetime(self.sentiment_df["trade_date"])
        price_daily = (
            self.price_df[["trade_date", "daily_return"]]
            .dropna()
            .drop_duplicates("trade_date")
        )
        self.merged = pd.merge(
            self.sentiment_df,
            price_daily,
            on="trade_date",
            how="inner"
        )
        print(f"[{self.ticker}] Merged rows: {len(self.merged):,} trading days with news.")
        return self.merged

   
    def pearson_correlation(self) -> dict:
        """Compute Pearson r between avg_sentiment and daily_return."""
        if self.merged is None:
            self.merge()
        df = self.merged.dropna(subset=["avg_sentiment", "daily_return"])

        r, p_value = stats.pearsonr(df["avg_sentiment"], df["daily_return"])
        n = len(df)

     
        z        = np.arctanh(r)
        se       = 1 / np.sqrt(n - 3)
        z_ci     = 1.96 * se
        ci_lower = np.tanh(z - z_ci)
        ci_upper = np.tanh(z + z_ci)

        result = {
            "ticker":          self.ticker,
            "pearson_r":       round(r, 6),
            "p_value":         round(p_value, 6),
            "significant":     p_value < 0.05,
            "n_observations":  n,
            "ci_95_lower":     round(ci_lower, 6),
            "ci_95_upper":     round(ci_upper, 6),
        }
        print(f"\n[{self.ticker}] Pearson r = {r:.4f} | p = {p_value:.4f} "
              f"| 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}] | n = {n}")
        return result

   
    def plot_scatter(self, save_path: str = None):
        """
        Scatter plot: daily sentiment score vs daily stock return,
        annotated with Pearson r and regression line.
        """
        if self.merged is None:
            self.merge()
        corr = self.pearson_correlation()
        df   = self.merged.dropna(subset=["avg_sentiment", "daily_return"])

        fig, ax = plt.subplots(figsize=(10, 7))

    
        for label, grp in df.groupby("sentiment_label"):
            ax.scatter(
                grp["avg_sentiment"], grp["daily_return"],
                label=label, color=COLORS.get(label, "steelblue"),
                alpha=0.65, edgecolors="white", linewidths=0.4, s=50
            )

      
        m, b = np.polyfit(df["avg_sentiment"], df["daily_return"], 1)
        x_range = np.linspace(df["avg_sentiment"].min(), df["avg_sentiment"].max(), 200)
        ax.plot(x_range, m * x_range + b, color="navy", linewidth=1.5,
                linestyle="--", label=f"Regression (slope={m:.3f})")

      
        ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)

        ax.set_xlabel("Average Daily Sentiment Score (VADER Compound)", fontsize=11)
        ax.set_ylabel("Daily Stock Return (%)", fontsize=11)
        ax.set_title(
            f"{self.ticker} – News Sentiment vs Daily Return\n"
            f"Pearson r = {corr['pearson_r']:.4f}  |  p = {corr['p_value']:.4f}  "
            f"| n = {corr['n_observations']:,}",
            fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=9)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_return_by_sentiment_category(self, save_path: str = None):
        """
        Bar chart: average daily return for Positive / Neutral / Negative days.
        """
        if self.merged is None:
            self.merge()

        category_means = (
            self.merged.groupby("sentiment_label")["daily_return"]
            .agg(["mean", "sem", "count"])
            .reindex(["Positive", "Neutral", "Negative"])
            .dropna()
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(
            category_means.index,
            category_means["mean"],
            yerr=category_means["sem"] * 1.96,
            capsize=6,
            color=[COLORS.get(l, "steelblue") for l in category_means.index],
            edgecolor="white",
            error_kw=dict(elinewidth=1.5, ecolor="navy"),
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

        for bar, (idx, row) in zip(bars, category_means.iterrows()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02 + row["sem"] * 1.96,
                f"{row['mean']:.3f}%\n(n={int(row['count'])})",
                ha="center", va="bottom", fontsize=10
            )

        ax.set_xlabel("Sentiment Category", fontsize=11)
        ax.set_ylabel("Average Daily Return (%)", fontsize=11)
        ax.set_title(
            f"{self.ticker} – Average Return by Sentiment Category\n"
            f"(Error bars = 95% CI)",
            fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_rolling_correlation(self, window: int = 30, save_path: str = None):
        """
        Rolling Pearson correlation to show how the relationship evolves over time.
        """
        if self.merged is None:
            self.merge()

        df = self.merged.sort_values("trade_date").dropna(
            subset=["avg_sentiment", "daily_return"]
        ).set_index("trade_date")

        rolling_r = (
            df["avg_sentiment"]
            .rolling(window)
            .corr(df["daily_return"])
        )

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(rolling_r.index, rolling_r.values, color="#9467bd", linewidth=1.4,
                label=f"{window}-day Rolling Pearson r")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.fill_between(rolling_r.index, rolling_r, 0,
                        where=rolling_r > 0, alpha=0.2, color="#2ca02c", label="Positive r")
        ax.fill_between(rolling_r.index, rolling_r, 0,
                        where=rolling_r < 0, alpha=0.2, color="#d62728", label="Negative r")
        ax.set_title(
            f"{self.ticker} – Rolling {window}-Day Sentiment–Return Correlation",
            fontsize=12, fontweight="bold"
        )
        ax.set_ylabel("Pearson r")
        ax.legend(fontsize=9)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

   
    @staticmethod
    def plot_correlation_heatmap(results: list, save_path: str = None):
        """
        Given a list of pearson_correlation() dicts, plot a summary bar chart.

        Parameters
        ----------
        results : list of dicts from pearson_correlation()
        """
        df = pd.DataFrame(results).sort_values("pearson_r", ascending=False)

        fig, ax = plt.subplots(figsize=(12, 6))
        bar_colors = ["#2ca02c" if r >= 0 else "#d62728" for r in df["pearson_r"]]
        bars = ax.bar(df["ticker"], df["pearson_r"], color=bar_colors, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8)

        for bar, (_, row) in zip(bars, df.iterrows()):
            sig = "*" if row["significant"] else ""
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.003 if bar.get_height() >= 0 else -0.008),
                f"{row['pearson_r']:.3f}{sig}",
                ha="center", fontsize=9,
            )

        ax.set_xlabel("Stock Ticker")
        ax.set_ylabel("Pearson r (Sentiment vs Return)")
        ax.set_title(
            "Sentiment–Return Pearson Correlation by Stock\n(* = p < 0.05)",
            fontsize=12, fontweight="bold"
        )
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
