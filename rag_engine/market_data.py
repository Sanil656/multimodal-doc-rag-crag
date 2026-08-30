import io
from typing import Dict, Any, Tuple, Optional
import pandas as pd


def fetch_live_ticker_data(
    ticker: str,
    period: str = "3mo",
    interval: str = "1d",
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Fetch live OHLCV data for any stock, crypto, or index from Yahoo Finance.
    Returns: (dataframe, error_message)
    """
    import yfinance as yf
    ticker = ticker.strip().upper()
    if not ticker:
        return None, "Please provide a valid ticker symbol."

    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval)
        if df.empty:
            return None, f"No market data found for symbol '{ticker}'. Please verify the ticker."
        
        # Clean dataframe
        df = df.dropna()
        if len(df) < 3:
            return None, f"Insufficient price data returned for '{ticker}'."
        return df, None
    except Exception as e:
        return None, f"Failed to fetch market data for '{ticker}': {str(e)}"


def create_interactive_candlestick_chart(df: pd.DataFrame, ticker: str):
    """
    Generate an interactive Plotly candlestick chart with volume bars.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.75, 0.25],
    )

    # 1. Candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        ),
        row=1,
        col=1,
    )

    # 2. 20-period Moving Average if enough data
    if len(df) >= 20:
        ma20 = df["Close"].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=ma20,
                name="20 MA",
                line=dict(color="#3b82f6", width=1.5),
            ),
            row=1,
            col=1,
        )

    # 3. Volume trace
    colors = ["#22c55e" if c >= o else "#ef4444" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.7,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"Live Candlestick Chart: {ticker.upper()}",
        xaxis_rangeslider_visible=False,
        height=480,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_candlestick_image_bytes(df: pd.DataFrame, ticker: str) -> bytes:
    """
    Render a clean, high-resolution candlestick chart as PNG bytes to feed to the Vision LLM.
    """
    import mplfinance as mpf

    # Create custom dark/light neutral style for vision model readability
    mc = mpf.make_marketcolors(
        up="#22c55e",
        down="#ef4444",
        edge="inherit",
        wick="inherit",
        volume="in",
    )
    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=mc,
        gridstyle=":",
        gridcolor="#444444",
    )

    buf = io.BytesIO()
    # Plot last 60 candles max for optimal candlestick pattern recognition
    plot_df = df.tail(60)
    
    mpf.plot(
        plot_df,
        type="candle",
        volume=True,
        title=f"{ticker.upper()} - Candlestick Price Action",
        style=style,
        savefig=dict(fname=buf, dpi=150, bbox_inches="tight", format="png"),
        figsize=(10, 6),
    )
    buf.seek(0)
    return buf.getvalue()


def calculate_market_summary(df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
    """
    Calculate summary stats (last price, change %, high/low range, volume).
    """
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2] if len(df) > 1 else last_row

    last_close = float(last_row["Close"])
    prev_close = float(prev_row["Close"])
    change = last_close - prev_close
    change_pct = (change / prev_close) * 100 if prev_close != 0 else 0.0

    return {
        "ticker": ticker.upper(),
        "last_price": round(last_close, 2),
        "change": round(change, 2),
        "change_pct": round(change_pct, 2),
        "high": round(float(df["High"].max()), 2),
        "low": round(float(df["Low"].min()), 2),
        "last_volume": int(last_row["Volume"]),
        "candles_count": len(df),
    }
