
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = sns.color_palette("muted")


class EDAAnalyzer:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._preprocess()

    def _preprocess(self):
      
        before = len(self.df)
        self.df.dropna(subset=["headline", "date"], inplace=True)
        after = len(self.df)
        print(f"[Preprocess] Dropped {before - after} rows with missing headline/date.")

        # Parse dates
        self.df["date"] = pd.to_datetime(self.df["date"], utc=True, errors="coerce")
        self.df.dropna(subset=["date"], inplace=True)

        # Derived columns
        self.df["headline_len"] = self.df["headline"].str.len()
        self.df["date_only"] = self.df["date"].dt.date
        self.df["hour"] = self.df["date"].dt.hour
        self.df["day_of_week"] = self.df["date"].dt.day_name()
        self.df["year_month"] = self.df["date"].dt.to_period("M")

        print(f"[Preprocess] Dataset ready: {len(self.df):,} rows.")

    # ── Descriptive Statistics ───────────────────────────────────────────────
    def descriptive_stats(self) -> pd.DataFrame:
        """Return summary stats for headline character length."""
        stats = self.df["headline_len"].describe().to_frame("headline_char_count")
        stats.loc["skewness"] = self.df["headline_len"].skew()
        stats.loc["kurtosis"] = self.df["headline_len"].kurtosis()
        return stats

    def plot_headline_length_distribution(self, save_path: str = None):
        """
        Figure 1 – Distribution of headline character lengths.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(
            self.df["headline_len"], bins=60, color=PALETTE[0], edgecolor="white", alpha=0.85
        )
        axes[0].axvline(
            self.df["headline_len"].mean(), color="crimson", linestyle="--", linewidth=1.8,
            label=f"Mean = {self.df['headline_len'].mean():.1f}"
        )
        axes[0].axvline(
            self.df["headline_len"].median(), color="navy", linestyle=":", linewidth=1.8,
            label=f"Median = {self.df['headline_len'].median():.1f}"
        )
        axes[0].set_title("Headline Character Length Distribution", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Character Count")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()

        # Box plot
        axes[1].boxplot(
            self.df["headline_len"], vert=False, patch_artist=True,
            boxprops=dict(facecolor=PALETTE[1], color="navy"),
            medianprops=dict(color="crimson", linewidth=2),
            whiskerprops=dict(color="navy"),
            capprops=dict(color="navy"),
        )
        axes[1].set_title("Headline Length Box Plot", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Character Count")
        axes[1].set_yticks([])

        plt.suptitle(
            "Nova Financial Solutions – EDA: Headline Length Analysis",
            fontsize=14, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    # ── Publisher Analysis ───────────────────────────────────────────────────
    def publisher_analysis(self, top_n: int = 20) -> pd.DataFrame:
        """Count articles per publisher and extract email domains."""
        pub_counts = (
            self.df["publisher"]
            .value_counts()
            .reset_index()
            .rename(columns={"publisher": "publisher", "count": "article_count"})
        )

        def extract_domain(name):
            match = re.search(r"@([\w.-]+)", str(name))
            return match.group(1) if match else None

        pub_counts["domain"] = pub_counts["publisher"].apply(extract_domain)
        return pub_counts.head(top_n)

    def plot_top_publishers(self, top_n: int = 15, save_path: str = None):
        """
        Figure 2 – Top publishers by article count.
        """
        pub_df = self.publisher_analysis(top_n)

        fig, ax = plt.subplots(figsize=(12, 7))
        bars = ax.barh(
            pub_df["publisher"].str[:35],
            pub_df["article_count"],
            color=sns.color_palette("Blues_r", len(pub_df)),
            edgecolor="white",
        )
        ax.bar_label(bars, padding=4, fontsize=9)
        ax.set_xlabel("Number of Articles", fontsize=11)
        ax.set_title(
            f"Top {top_n} Most Active Publishers", fontsize=13, fontweight="bold"
        )
        ax.invert_yaxis()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    # ── Time Series Analysis ─────────────────────────────────────────────────
    def news_volume_over_time(self) -> pd.DataFrame:
        """Daily article counts."""
        return (
            self.df.groupby("date_only")
            .size()
            .reset_index(name="article_count")
            .sort_values("date_only")
        )

    def detect_spikes(self, threshold_std: float = 2.5) -> pd.DataFrame:
        """Flag days where article volume exceeds mean + threshold_std * std."""
        daily = self.news_volume_over_time()
        mean = daily["article_count"].mean()
        std = daily["article_count"].std()
        daily["is_spike"] = daily["article_count"] > (mean + threshold_std * std)
        return daily[daily["is_spike"]]

    def plot_news_volume_timeseries(self, save_path: str = None):
        """
        Figure 3 – News volume over time with spike highlights.
        """
        daily = self.news_volume_over_time()
        spikes = self.detect_spikes()

        daily["date_only"] = pd.to_datetime(daily["date_only"])
        spikes["date_only"] = pd.to_datetime(spikes["date_only"])

        fig, ax = plt.subplots(figsize=(16, 6))

        ax.fill_between(
            daily["date_only"], daily["article_count"],
            alpha=0.35, color=PALETTE[0], label="Daily Volume"
        )
        ax.plot(daily["date_only"], daily["article_count"], color=PALETTE[0], linewidth=0.8)
        ax.scatter(
            spikes["date_only"], spikes["article_count"],
            color="crimson", zorder=5, s=45, label="Volume Spike", marker="^"
        )
        # Rolling 7-day average
        rolling = daily.set_index("date_only")["article_count"].rolling(7).mean()
        ax.plot(rolling.index, rolling.values, color="navy", linewidth=1.5,
                linestyle="--", label="7-day Rolling Avg")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=30)
        ax.set_title(
            "Financial News Publication Volume Over Time",
            fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Articles Published")
        ax.legend(loc="upper left")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_hourly_distribution(self, save_path: str = None):
        """Bar chart – at what hour of the day are articles published?"""
        hourly = self.df["hour"].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(hourly.index, hourly.values, color=PALETTE[2], edgecolor="white")
        ax.set_xticks(range(0, 24))
        ax.set_xlabel("Hour of Day (UTC-4)")
        ax.set_ylabel("Number of Articles")
        ax.set_title("News Publication by Hour of Day", fontsize=13, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def top_tfidf_keywords(self, top_n: int = 20) -> pd.DataFrame:
        """Extract top N keywords by mean TF-IDF score."""
        headlines = self.df["headline"].dropna().tolist()
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
            min_df=5,
        )
        tfidf_matrix = vectorizer.fit_transform(headlines)
        mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        keywords = pd.DataFrame(
            {"keyword": vectorizer.get_feature_names_out(), "tfidf_score": mean_scores}
        ).sort_values("tfidf_score", ascending=False)
        return keywords.head(top_n)

    def plot_top_keywords(self, top_n: int = 20, save_path: str = None):
        """
        Figure 4 – Top TF-IDF keywords in financial headlines.
        """
        kw_df = self.top_tfidf_keywords(top_n)

        fig, ax = plt.subplots(figsize=(12, 7))
        bars = ax.barh(
            kw_df["keyword"][::-1],
            kw_df["tfidf_score"][::-1],
            color=sns.color_palette("YlOrRd_r", top_n),
            edgecolor="white",
        )
        ax.set_xlabel("Mean TF-IDF Score", fontsize=11)
        ax.set_title(
            f"Top {top_n} Recurring Financial News Keywords (TF-IDF)",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_publication_heatmap(self, save_path: str = None):
        """
        Figure 5 – Heatmap of publication volume by day-of-week × hour.
        """
        pivot = (
            self.df.groupby(["day_of_week", "hour"])
            .size()
            .unstack(fill_value=0)
        )
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        pivot = pivot.reindex([d for d in day_order if d in pivot.index])

        fig, ax = plt.subplots(figsize=(16, 5))
        sns.heatmap(
            pivot, cmap="YlOrRd", ax=ax,
            linewidths=0.3, cbar_kws={"label": "Article Count"}
        )
        ax.set_title(
            "Publication Frequency Heatmap (Day of Week × Hour)",
            fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("Hour of Day (UTC-4)")
        ax.set_ylabel("Day of Week")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def stock_coverage_summary(self, top_n: int = 15) -> pd.DataFrame:
        """Count articles per stock ticker."""
        return (
            self.df["stock"]
            .value_counts()
            .head(top_n)
            .reset_index()
            .rename(columns={"stock": "ticker", "count": "article_count"})
        )
