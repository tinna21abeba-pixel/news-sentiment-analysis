"""
technical_indicators.py
=======================
Nova Financial Solutions – Task 2
Compute technical indicators with TA-Lib and additional metrics with PyNance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("[WARNING] TA-Lib not installed. Using pandas-based fallback for indicators.")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


class TechnicalAnalyzer:
    """
    Loads and enriches a stock price DataFrame with technical indicators.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Date, Open, High, Low, Close, Adj Close, Volume
    ticker : str
        Stock ticker label used in plot titles.
    """

    def __init__(self, df: pd.DataFrame, ticker: str = "STOCK"):
        self.ticker = ticker
        self.df = self._prepare(df.copy())

    # ── Data Preparation ─────────────────────────────────────────────────────
    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse dates, sort, and forward-fill small gaps."""
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
        df.set_index("Date", inplace=True)

        numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        missing_before = df.isnull().sum().sum()
        df.ffill(inplace=True)
        df.dropna(subset=["Close", "Adj Close"], inplace=True)
        missing_after = df.isnull().sum().sum()

        print(f"[{self.ticker}] Data range: {df.index.min().date()} → {df.index.max().date()}")
        print(f"[{self.ticker}] Rows: {len(df):,} | Missing values filled: {missing_before - missing_after}")

        return df

    # ── Indicator Computation ────────────────────────────────────────────────
    def compute_all_indicators(self) -> pd.DataFrame:
        """Compute SMA, EMA, RSI, MACD, Bollinger Bands, and ATR."""
        close = self.df["Close"].values.astype(float)
        high  = self.df["High"].values.astype(float)
        low   = self.df["Low"].values.astype(float)

        if TALIB_AVAILABLE:
            self.df["SMA_20"]  = talib.SMA(close, timeperiod=20)
            self.df["SMA_50"]  = talib.SMA(close, timeperiod=50)
            self.df["EMA_20"]  = talib.EMA(close, timeperiod=20)
            self.df["EMA_50"]  = talib.EMA(close, timeperiod=50)
            self.df["RSI_14"]  = talib.RSI(close, timeperiod=14)
            macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            self.df["MACD"]        = macd
            self.df["MACD_Signal"] = signal
            self.df["MACD_Hist"]   = hist
            upper, middle, lower   = talib.BBANDS(close, timeperiod=20)
            self.df["BB_Upper"]    = upper
            self.df["BB_Middle"]   = middle
            self.df["BB_Lower"]    = lower
            self.df["ATR_14"]      = talib.ATR(high, low, close, timeperiod=14)
        else:
            # Pandas fallback
            self.df["SMA_20"] = self.df["Close"].rolling(20).mean()
            self.df["SMA_50"] = self.df["Close"].rolling(50).mean()
            self.df["EMA_20"] = self.df["Close"].ewm(span=20, adjust=False).mean()
            self.df["EMA_50"] = self.df["Close"].ewm(span=50, adjust=False).mean()
            self.df["RSI_14"] = self._pandas_rsi(self.df["Close"], 14)
            self.df["MACD"], self.df["MACD_Signal"], self.df["MACD_Hist"] = \
                self._pandas_macd(self.df["Close"])
            self.df["BB_Upper"], self.df["BB_Middle"], self.df["BB_Lower"] = \
                self._pandas_bbands(self.df["Close"], 20)
            self.df["ATR_14"] = self._pandas_atr(self.df, 14)

        # Daily Return
        self.df["Daily_Return"] = self.df["Adj Close"].pct_change() * 100
        # Cumulative Return
        self.df["Cum_Return"] = (1 + self.df["Adj Close"].pct_change()).cumprod() - 1

        print(f"[{self.ticker}] Indicators computed successfully.")
        return self.df

    # ── Pandas Fallback Indicators ────────────────────────────────────────────
    @staticmethod
    def _pandas_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _pandas_macd(series: pd.Series, fast=12, slow=26, signal=9):
        ema_fast   = series.ewm(span=fast, adjust=False).mean()
        ema_slow   = series.ewm(span=slow, adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram  = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def _pandas_bbands(series: pd.Series, period: int = 20):
        middle = series.rolling(period).mean()
        std    = series.rolling(period).std()
        return middle + 2 * std, middle, middle - 2 * std

    @staticmethod
    def _pandas_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df["High"], df["Low"], df["Close"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    # ── PyNance / Additional Metrics ─────────────────────────────────────────
    def compute_pynance_metrics(self) -> dict:
        """
        Compute additional financial metrics equivalent to PyNance outputs.
        Returns Sharpe Ratio, Sortino Ratio, Max Drawdown, Volatility.
        """
        returns = self.df["Daily_Return"].dropna() / 100  # decimal

        # Annualised Volatility
        vol = returns.std() * np.sqrt(252)

        # Sharpe Ratio (assumes risk-free rate = 0 for simplicity)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

        # Sortino Ratio
        downside = returns[returns < 0].std() * np.sqrt(252)
        sortino  = (returns.mean() * 252) / downside if downside != 0 else np.nan

        # Maximum Drawdown
        cum_returns  = (1 + returns).cumprod()
        rolling_max  = cum_returns.cummax()
        drawdown     = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Calmar Ratio
        annualised_return = returns.mean() * 252
        calmar = annualised_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

        metrics = {
            "Annualised Volatility": f"{vol:.4f}",
            "Sharpe Ratio":          f"{sharpe:.4f}",
            "Sortino Ratio":         f"{sortino:.4f}",
            "Max Drawdown":          f"{max_drawdown:.4f}",
            "Calmar Ratio":          f"{calmar:.4f}",
            "Total Trading Days":    len(returns),
        }
        print(f"\n[{self.ticker}] PyNance-equivalent Financial Metrics:")
        for k, v in metrics.items():
            print(f"  {k:30s}: {v}")
        return metrics

    # ── Multi-Panel Visualization ────────────────────────────────────────────
    def plot_indicators(self, save_path: str = None):
        """
        Publication-quality 4-panel chart:
          Panel 1 – Closing Price + SMA20 + EMA50 + Bollinger Bands
          Panel 2 – Volume
          Panel 3 – RSI (14) with 30/70 bands
          Panel 4 – MACD histogram + signal
        """
        df = self.df.dropna(subset=["SMA_20", "RSI_14", "MACD"])

        fig = plt.figure(figsize=(18, 14))
        gs  = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1.5, 1.5], hspace=0.05)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax4 = fig.add_subplot(gs[3], sharex=ax1)

        # ── Panel 1: Price + MAs + Bollinger ─────────────────────────────────
        ax1.plot(df.index, df["Close"],    color="#1f77b4", linewidth=1.4,  label="Close Price", zorder=3)
        ax1.plot(df.index, df["SMA_20"],   color="#ff7f0e", linewidth=1.2,  linestyle="--", label="SMA 20")
        ax1.plot(df.index, df["EMA_50"],   color="#2ca02c", linewidth=1.2,  linestyle="-.", label="EMA 50")
        ax1.fill_between(df.index, df["BB_Upper"], df["BB_Lower"],
                         alpha=0.12, color="gray", label="Bollinger Bands (20,2)")
        ax1.plot(df.index, df["BB_Upper"], color="gray", linewidth=0.6, linestyle=":")
        ax1.plot(df.index, df["BB_Lower"], color="gray", linewidth=0.6, linestyle=":")
        ax1.set_ylabel("Price (USD)", fontsize=10)
        ax1.set_title(f"{self.ticker} – Technical Indicator Dashboard", fontsize=14, fontweight="bold")
        ax1.legend(loc="upper left", fontsize=9, ncol=4)
        ax1.tick_params(labelbottom=False)

        # ── Panel 2: Volume ───────────────────────────────────────────────────
        colors = ["#d62728" if r < 0 else "#2ca02c"
                  for r in df["Daily_Return"].fillna(0)]
        ax2.bar(df.index, df["Volume"], color=colors, alpha=0.7, width=1)
        ax2.set_ylabel("Volume", fontsize=10)
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M")
        )
        ax2.tick_params(labelbottom=False)

        # ── Panel 3: RSI ──────────────────────────────────────────────────────
        ax3.plot(df.index, df["RSI_14"], color="#9467bd", linewidth=1.3, label="RSI (14)")
        ax3.axhline(70, color="crimson", linestyle="--", linewidth=1, alpha=0.8, label="Overbought (70)")
        ax3.axhline(30, color="green",   linestyle="--", linewidth=1, alpha=0.8, label="Oversold (30)")
        ax3.fill_between(df.index, 70, df["RSI_14"].clip(upper=100),
                         where=df["RSI_14"] > 70, alpha=0.2, color="crimson")
        ax3.fill_between(df.index, df["RSI_14"].clip(lower=0), 30,
                         where=df["RSI_14"] < 30, alpha=0.2, color="green")
        ax3.set_ylim(0, 100)
        ax3.set_ylabel("RSI (14)", fontsize=10)
        ax3.legend(loc="upper left", fontsize=8, ncol=3)
        ax3.tick_params(labelbottom=False)

        # ── Panel 4: MACD ─────────────────────────────────────────────────────
        ax4.plot(df.index, df["MACD"],        color="#1f77b4", linewidth=1.2, label="MACD")
        ax4.plot(df.index, df["MACD_Signal"], color="#ff7f0e", linewidth=1.2, label="Signal")
        hist_colors = ["#d62728" if v < 0 else "#2ca02c"
                       for v in df["MACD_Hist"].fillna(0)]
        ax4.bar(df.index, df["MACD_Hist"], color=hist_colors, alpha=0.6, width=1, label="Histogram")
        ax4.axhline(0, color="black", linewidth=0.8, linestyle="-")
        ax4.set_ylabel("MACD", fontsize=10)
        ax4.legend(loc="upper left", fontsize=8, ncol=3)

        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha="right")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_cumulative_return(self, save_path: str = None):
        """Cumulative return over the analysis period."""
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(self.df.index, self.df["Cum_Return"] * 100,
                color="#1f77b4", linewidth=1.5)
        ax.fill_between(self.df.index, self.df["Cum_Return"] * 100,
                        alpha=0.2, color="#1f77b4")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{self.ticker} – Cumulative Return (%)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Cumulative Return (%)")
        ax.set_xlabel("Date")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
