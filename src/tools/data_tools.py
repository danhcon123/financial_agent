"""
LangChain tools for data fetching and storage
"""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
from src.data.validators import OHLCVValidator
from src.data.storage import DuckDBStorage
from src.models.market_data import PriceDataSummary
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# Tool Input Schemas
# ============================================================================

class FetchPriceDataInput(BaseModel):
    """Input schema for price data fetching tool"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL, TSLA)")
    days: int = Field(90, ge=1, le=365, description="Number of days of historical data")

# ============================================================================
# Tool Fuctions
# ============================================================================

def fetch_and_store_price_data(ticker: str, days: int = 90) -> dict:
    """
    Fetch historical price data from Yahoo Finance, validate, and store in DuckDB
    
    Args:
        ticker: Stock symbol
        days: Number of calendar days of history

    Returns:
        Dictionary with summary statistics and status
    """
    try:
        # Fetch data
        fetcher = YahooFinanceFetcher()
        rows, summary = fetcher.fetch(ticker, days)

        # Validate data
        validator = OHLCVValidator()
        validation = validator.validate(rows)
        
        if not validation.is_valid:
            logger.error(f"Validation failed for {ticker}: {validation.errors}")
            return {
                "success": False,
                "ticker": ticker,
                "error": f"Validation failed: {', '.join(validation.errors[:3])}"
            }
        
        # Store in database
        storage = DuckDBStorage()
        rows_stored = storage.store(rows, replace=True)
        storage.close()
    

        # Return summary
        return {
            "success": True,
            "ticker": summary.ticker,
            "rows_fetched": summary.row_count,
            "rows_stored": rows_stored,
            "start_date": str(summary.start_date),
            "end_date": str(summary.end_date),
            "latest_close": summary.price_change_pct,
            "price_change_pct": summary.price_change_pct,
            "avg_volume": summary.avg_volume,
            "validation_warnings": len(validation.warnings),
            "evidence_claim": summary.to_evidence_claim()
        }
    
    except Exception as e:
        logger.exception(f"Failed to fetch/store price data for {ticker}")
        return {
            "success": False,
            "ticker": ticker,
            "error": str(e)
        }

# ============================================================================
# Tool Definitions
# ============================================================================

fetch_price_tool = StructuredTool.from_function(
    func=fetch_and_store_price_data,
    name="fetch_stock_price_data",
    description=(
        "Fetches historical OHLCV (Open, High, Low, Close, Volume) price data "
        "for a stock ticker from Yahoo Finance. Validates data quality and "
        "stores in local database. Returns summary statistics and evidence claim."
        ),
        args_schema=FetchPriceDataInput
)

# List of all data tools (for easy import)
DATA_TOOLS = [fetch_price_tool]