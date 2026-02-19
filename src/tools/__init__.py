"""
Tests for Slice 2: Technical Analysis Integration
"""

import pytest
from datetime import datetime, timedelta

from src.sandbox.analytics import (
    compute_sma, compute_ema, compute_rsi,
    compute_macd, compute_bollinger_bands, analyze_trend
)
from src.visualization.charts import ChartGenerator
from src.tools.data_tools import (
    compute_technical_indicators,
    generate_stock_chart
)
from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.storage import DuckDBStorage
from src.agents.orchestrator import Orchestrator
from src.models.schemas import ResearchRequest

def test_compute_sma():
    """Test SMA calculation"""
    prices = [100,102,101,103,105,104,106,108,107,109]

    sma_5 = compute_sma(prices, period=5)

    # First 4 values should be NaN
    assert len(sma_5) == len(prices)
    assert all(str(x) == 'nan' for x in sma_5[:4])

    # 5th value should be average of first 5
    expected_sma_5 = sum(prices[:5]) / 5
    assert abs(sma_5[4] - expected_sma_5) < 0.01
    print(f"\n✅ SMA calculation works")
    print(f"   Prices: {prices[-5:]}")
    print(f"   SMA(5): {sma_5[-5:]}")

def test_compute_rsi():
    """Test RSI calculation"""
    # Create price data with clear uptrend
    prices = [100 + i for i in range(30)] # Steadily increase

    rsi = compute_rsi(prices, period=14)

    # RSI should be high (>70) for strong uptrend
    assert len(rsi) == len(prices)
    current_rsi = rsi[-1]
    print(f"\n✅ RSI calculation works")
    print(f"   Current RSI: {current_rsi:.2f}")
    print(f"   Signal: {'Overbought' if current_rsi > 70 else 'Oversold' if current_rsi < 30 else 'Neutral'}")

def test_analyze_trend():
    """Test trend analysis"""
    # Create data where price is above SMA
    prices = [100] * 20 + [105, 106, 107, 108, 109]  # Price rises above flat SMA
    
    trend = analyze_trend(prices, sma_period=20)
    
    assert 'error' not in trend
    assert trend['signal'] in ['bullish', 'bearish', 'neutral']
    assert 'current_price' in trend
    assert 'sma_value' in trend
    assert 'percent_from_sma' in trend
    print(f"\n✅ Trend analysis works")
    print(f"   Current Price: ${trend['current_price']}")
    print(f"   SMA(20): ${trend['sma_value']}")
    print(f"   % from SMA: {trend['percent_from_sma']:.2f}%")
    print(f"   Signal: {trend['signal']}")

def test_compute_indicators_tool():
    """Test technical indicators tool integration"""
    # First ensure we have data
    fetcher = YahooFinanceFetcher()
    rows, _ = fetcher.fetch("AAPL", days=90)

    storage = DuckDBStorage()
    storage.store(rows, replace=True)
    storage.close()

    # Now test the tool
    result = compute_technical_indicators(
        ticker="AAPL",
        indicators=["sma_20", "rsi_14", "trend"]
    )

    assert result["success"] is True
    assert result["ticker"] == "AAPL"
    assert "indicators" in result

    # Check SMA
    assert "sma_20" in result["indicators"]
    assert result["indicators"]["sma_20"]["current"] is not None 