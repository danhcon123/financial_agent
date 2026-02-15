import pytest
from datetime import datetime, timedelta

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.validators import OHLCVValidator
from src.data.storage import DuckDBStorage
from src.tools.data_tools import fetch_and_store_price_data
from src.agents.orchestrator import Orchestrator
from src.models.schemas import ResearchRequest


@pytest.fixture
def test_db_path(tmp_path):
    """Temporary database for testing"""
    return str(tmp_path / "test_market_data.db")


def test_yahoo_finance_fetcher():
    """Test basic price data fetching"""
    fetcher = YahooFinanceFetcher()
    
    # Fetch Apple data for last 30 days
    rows, summary = fetcher.fetch("AAPL", days=30)
    
    # Assertions
    assert len(rows) > 0
    assert summary.ticker == "AAPL"
    assert summary.row_count == len(rows)
    assert summary.latest_close > 0
    assert summary.avg_volume > 0
    
    # Check first row structure
    first_row = rows[0]
    assert first_row.ticker == "AAPL"
    assert first_row.open > 0
    assert first_row.high >= first_row.low
    assert first_row.volume >= 0
    
    print(f"\n✅ Fetched {len(rows)} days of AAPL data")
    print(f"   Latest close: ${summary.latest_close:.2f}")
    print(f"   Price change: {summary.price_change_pct:+.2f}%")


def test_ohlcv_validator():
    """Test data validation"""
    fetcher = YahooFinanceFetcher()
    rows, _ = fetcher.fetch("MSFT", days=30)
    
    validator = OHLCVValidator(max_price_change_pct=50.0)
    result = validator.validate(rows)
    
    assert result.rows_checked == len(rows)
    assert result.rows_passed > 0
    
    # Real market data should be valid
    if not result.is_valid:
        print(f"\n⚠️ Validation errors: {result.errors}")
    else:
        print(f"\n✅ Validation passed: {result.rows_passed}/{result.rows_checked} rows")
        if result.warnings:
            print(f"   Warnings: {len(result.warnings)}")


def test_duckdb_storage(test_db_path):
    """Test database storage and retrieval"""
    fetcher = YahooFinanceFetcher()
    rows, summary = fetcher.fetch("TSLA", days=30)
    
    # Store data
    storage = DuckDBStorage(test_db_path)
    rows_stored = storage.store(rows, replace=True)
    
    assert rows_stored == len(rows)
    
    # Retrieve data
    retrieved_rows = storage.fetch("TSLA", limit=10)
    assert len(retrieved_rows) <= 10
    assert retrieved_rows[0].ticker == "TSLA"
    
    # Get latest
    latest = storage.get_latest("TSLA")
    assert latest is not None
    assert latest.close == summary.latest_close
    
    storage.close()
    
    print(f"\n✅ Stored and retrieved {rows_stored} TSLA rows")
    print(f"   Latest close: ${latest.close:.2f}")


def test_tool_integration():
    """Test LangChain tool wrapper"""
    result = fetch_and_store_price_data("NVDA", days=60)
    
    assert result["success"] is True
    assert result["ticker"] == "NVDA"
    assert result["rows_fetched"] > 0
    assert result["latest_close"] > 0
    assert "evidence_claim" in result
    
    print(f"\n✅ Tool fetched {result['rows_fetched']} NVDA rows")
    print(f"   Evidence: {result['evidence_claim'][:100]}...")


@pytest.mark.asyncio
async def test_orchestrator_with_real_data():
    """Test full orchestrator with real price data (Slice 1 integration)"""
    orchestrator = Orchestrator()
    
    request = ResearchRequest(
        query="Analyze recent price action and provide investment recommendation",
        ticker="AAPL",
        horizon="3 months",
        risk_profile="moderate",
        max_iterations=1
    )
    
    result = await orchestrator.run(request)
    
    # Assertions
    assert result.ok is True
    assert len(result.evidence) > 0
    
    # Check that we got REAL data (not stub)
    price_evidence = result.evidence[0]
    assert price_evidence.source == "yahoo_finance"
    assert price_evidence.confidence > 0.8  # High confidence
    assert "closed at $" in price_evidence.claim  # Real price data
    
    # Analyst should reference actual prices
    assert result.analyst_output is not None
    assert "$" in result.analyst_output.thesis  # Should mention dollar amounts
    
    print(f"\n{'='*60}")
    print("SLICE 1 INTEGRATION TEST RESULTS")
    print(f"{'='*60}")
    print(f"Run ID: {result.run_id}")
    print(f"Evidence items: {len(result.evidence)}")
    print(f"\nPrice Evidence (E1):")
    print(f"  {price_evidence.claim}")
    print(f"\nAnalyst Thesis (first 200 chars):")
    print(f"  {result.analyst_output.thesis[:200]}...")
    print(f"\nRecommendation: {result.analyst_output.recommended_action}")
    print(f"Critic Assessment: {result.critic_output.assessment if result.critic_output else 'N/A'}")
    print(f"{'='*60}")