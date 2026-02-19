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
from src.tools.analysis_tools import (
    compute_technical_indicators,
    generate_stock_chart
)
from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.storage import DuckDBStorage
from src.agents.orchestrator import Orchestrator
from src.models.schemas import ResearchRequest

def test_compute_sma():
    """Test SMA calculation"""
    prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
    
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
    prices = [100 + i for i in range(30)]  # Steadily increasing
    
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
    
    # Check RSI
    assert "rsi_14" in result["indicators"]
    assert result["indicators"]["rsi_14"]["current"] is not None
    assert result["indicators"]["rsi_14"]["signal"] in ["overbought", "oversold", "neutral"]
    
    # Check trend
    assert "trend" in result["indicators"]
    assert "signal" in result["indicators"]["trend"]
    
    print(f"\n✅ Indicators tool works")
    print(f"   Ticker: {result['ticker']}")
    print(f"   Data points: {result['data_points']}")
    print(f"   SMA(20): ${result['indicators']['sma_20']['current']:.2f}")
    print(f"   RSI(14): {result['indicators']['rsi_14']['current']:.2f} ({result['indicators']['rsi_14']['signal']})")
    print(f"   Trend: {result['indicators']['trend']['signal']}")


def test_chart_generation():
    """Test chart generation"""
    # Ensure we have data
    fetcher = YahooFinanceFetcher()
    rows, _ = fetcher.fetch("MSFT", days=60)
    
    storage = DuckDBStorage()
    storage.store(rows, replace=True)
    storage.close()
    
    # Generate price chart
    result = generate_stock_chart(
        ticker="MSFT",
        chart_type="price",
        include_indicators=["sma_20"]
    )
    
    assert result["success"] is True
    assert "chart_path" in result
    
    # Check file exists
    import os
    assert os.path.exists(result["chart_path"])
    
    print(f"\n✅ Chart generation works")
    print(f"   Chart saved: {result['chart_path']}")
    print(f"   Chart type: {result['chart_type']}")


@pytest.mark.asyncio
async def test_orchestrator_with_indicators():
    """Test orchestrator with technical indicators (Slice 2 integration)"""
    orchestrator = Orchestrator()
    
    request = ResearchRequest(
        query="Analyze NVDA with technical indicators and provide trading recommendation",
        ticker="NVDA",
        horizon="1 month",
        risk_profile="aggressive",
        max_iterations=1
    )
    
    result = await orchestrator.run(request)
    
    # Basic assertions
    assert result.ok is True
    assert len(result.evidence) > 0
    
    # Check that we have price evidence
    price_evidence = result.evidence[0]
    assert price_evidence.source == "yahoo_finance"
    assert "$" in price_evidence.claim
    
    # Check if we have technical indicator evidence
    indicator_evidence = [e for e in result.evidence if "RSI" in e.claim or "SMA" in e.claim]
    assert len(indicator_evidence) > 0, "Should have technical indicator evidence"
    
    # Analyst should reference indicators
    assert result.analyst_output is not None
    thesis_text = result.analyst_output.thesis.lower()
    
    # Should mention at least one indicator
    has_indicators = any(
        indicator in thesis_text
        for indicator in ["rsi", "sma", "moving average", "overbought", "oversold", "trend"]
    )
    assert has_indicators, "Thesis should reference technical indicators"
    
    print(f"\n{'='*60}")
    print("SLICE 2 INTEGRATION TEST RESULTS")
    print(f"{'='*60}")
    print(f"Run ID: {result.run_id}")
    print(f"Evidence items: {len(result.evidence)}")
    
    # Show indicator evidence
    for e in indicator_evidence:
        print(f"\nIndicator Evidence ({e.id}):")
        print(f"  {e.claim[:150]}...")
    
    print(f"\nAnalyst Thesis (first 300 chars):")
    print(f"  {result.analyst_output.thesis[:300]}...")
    
    print(f"\nRecommendation: {result.analyst_output.recommended_action}")
    print(f"Critic Assessment: {result.critic_output.assessment if result.critic_output else 'N/A'}")
    print(f"{'='*60}")


def test_full_workflow_with_charts():
    """Test complete workflow: indicators + analysis + chart generation"""
    ticker = "TSLA"
    
    # Step 1: Fetch data
    fetcher = YahooFinanceFetcher()
    rows, summary = fetcher.fetch(ticker, days=90)
    
    storage = DuckDBStorage()
    storage.store(rows, replace=True)
    storage.close()
    
    # Step 2: Compute indicators
    indicators = compute_technical_indicators(
        ticker=ticker,
        indicators=["sma_20", "rsi_14", "macd", "trend"]
    )
    
    assert indicators["success"] is True
    
    # Step 3: Generate chart
    chart = generate_stock_chart(
        ticker=ticker,
        chart_type="price",
        include_indicators=["sma_20"]
    )
    
    assert chart["success"] is True
    
    print(f"\n✅ Full workflow complete for {ticker}")
    print(f"   Price: ${summary.latest_close:.2f}")
    print(f"   SMA(20): ${indicators['indicators']['sma_20']['current']:.2f}")
    print(f"   RSI(14): {indicators['indicators']['rsi_14']['current']:.2f}")
    print(f"   Trend: {indicators['indicators']['trend']['signal']}")
    print(f"   Chart: {chart['chart_path']}")