"""
Chart generation for financial data visualization.
"""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import date as date_type
import numpy as np

from src.data.storage import DuckDBStorage
from src.sandbox.analytics import (
    compute_sma, compute_ema, compute_rsi,
    compute_macd, compute_bollinger_bands, analyze_trend
)
from src.visualization.charts import ChartGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# Tool Input Schemas
# ============================================================================

class ComputeIndicatorsInput(BaseModel):
    """Input schema for computing technical indicators"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    indicators: List[str] = Field(
        ...,
        description="List of indicators to compute: sma_20, ema_20, rsi_14, macd, bbands"
    )

class GenerateChartInput(BaseModel):
    """Input schema for chart generation"""
    ticker: str = Field(..., description="Stock ticker symbol")
    chart_type: str = Field(
        "price",
        description="Type of chart: 'price' or 'rsi'"
    )
    include_indicators: List[str] = Field(
        default_factory=list,
        description="Indicators to overlay on price chart: sma_20, ema_20, bbands"
    )


# ============================================================================
# Tool Functions
# ============================================================================
def compute_technical_indicators(
    ticker: str,
    indicators: List[str]
) -> Dict[str, Any]:
    """
    Compute technical indicators for a stock.
    
    Args: 
        ticker: Stock ticker symbol
        indicators: List of indicators (e.g., ['sma_20', 'rsi_14', 'macd'])

    Returns:
        Dictionary with computed indicators and analysis
    """
    try:
        # Fetch price data from DuckDB
        storage = DuckDBStorage()
        rows = storage.fetch(ticker, limit=252) # ~1 year of data
        storage.close()

        if not rows:
            return{
                "success": False,
                "error": f"No data found for {ticker}"
            }
        
        # Sort by date ascending (oldest first)
        rows = sorted(rows, key=lambda x: x.date)

        dates = [r.date for r in rows]
        closes = [r.close for r in rows]

        result = {
            "success": True,
            "ticker": ticker,
            "data_points": len(rows),
            "date_range": f"{dates[0]} to {dates[-1]}",
            "indicators": {}
        }

        # Compute requested indicators
        for indicator in indicators:
            indicator = indicator.lower().strip()

            if indicator == 'sma_20':
                sma = compute_sma(closes, period=20)
                result['indicators']['sma_20'] = {
                    'current': sma[-1] if not np.isnan(sma[-1]) else None,
                    'values': sma[-10:] # Last 10 value
                }
            
            elif indicator == 'ema_20':
                ema = compute_ema(closes, period=20)
                result['indicators']['ema_20'] = {
                    'current': ema[-1] if not np.isnan(ema[-1]) else None,
                    'values': ema[-10:]
                }
            
            elif indicator == 'rsi_14':
                rsi = compute_rsi(closes, period=14)
                current_rsi = rsi[-1]
                result['indicators']['rsi_14'] = {
                    'current': current_rsi if not np.isnan(current_rsi) else None,
                    'signal': 'overbought' if current_rsi > 70 else 'oversold' if current_rsi < 30 else 'neutral',
                    'values': rsi[-10:]
                }
            
            elif indicator == 'macd':
                macd_data = compute_macd(closes)
                result['indicators']['macd'] = {
                    'macd_current': macd_data['macd'][-1],
                    'signal_current': macd_data['signal'][-1],
                    'histogram_current': macd_data['histogram'][-1]
                }

            elif indicator == 'bbands':
                bbands = compute_bollinger_bands(closes, period=20)
                result['indicators']['bbands'] = {
                    'upper': bbands['upper'][-1],
                    'middle': bbands['middle'][-1],
                    'lower': bbands['lower'][-1]
                }

            elif indicator == 'trend':
                trend_analysis = analyze_trend(closes, sma_period=20)
                result['indicators']['trend'] = trend_analysis

        return result
    
    except Exception as e:
        logger.exception(f"Failed to compute indicators for {ticker}")
        return {
            "success": False,
            "error": str(e)
        }
    
def generate_stock_chart(
    ticker: str,
    chart_type: str = "price",
    include_indicators: List[str] = []
) -> Dict[str, Any]:
    """
    Generate a chart for a stock.

    Args:
        ticker: Stock ticker symbol
        chart_type: Type of chart ('price' or 'rsi')
        included_indicators: Indicators to overlay

    Returns:
        Dictionary with chart path and metadata
    """
    try:
        # Fetch price data
        storage = DuckDBStorage()
        rows = storage.fetch(ticker, limit=90) # Last 90 days
        storage.close()

        if not rows:
            return {
                "success": False,
                "error": f"No data found for {ticker}"
            }
        
        # Sort by date
        rows = sorted(rows, key=lambda x: x.date)
        dates = [r.date for r in rows]
        closes = [r.close for r in rows]

        chart_gen = ChartGenerator()

        if chart_type == "rsi":
            # Generate RSI chart
            rsi_values = compute_rsi(closes, period=14)
            chart_path = chart_gen.generate_rsi_chart(dates, rsi_values, ticker)

            return {
                "success": True,
                "chart_path": chart_path,
                "chart_type" : "rsi",
                "ticker": ticker,
            }
        
        else: # price chart
            # Compute indicators to overlay
            indicators_dict = {}

            for ind in include_indicators:
                ind = ind.lower().strip()
                if ind == 'sma_20':
                    indicators_dict['SMA(20)'] = compute_sma(closes, 20)
                elif ind == 'ema_20':
                    indicators_dict['EMA(20)'] = compute_ema(closes, 20)
                elif ind == 'bbands':
                    bbands = compute_bollinger_bands(closes, 20)
                    indicators_dict['BB Upper'] = bbands['upper']
                    indicators_dict['BB Lower'] = bbands['lower']

            chart_path = chart_gen.generate_price_chart(
                dates, closes, ticker, indicators=indicators_dict
            )

            return {
                "success": True,
                "chart_path": chart_path,
                "chart_type": "price",
                "ticker": ticker,
                "indicators": list(indicators_dict.keys())
            }
    except Exception as e:
        logger.exception(f"Failed to generate chart for {ticker}")
        return {
            "success": False,
            "error": str(e)
        }

# ============================================================================
# Tool Definitions
# ============================================================================

compute_indicators_tool = StructuredTool.from_function(
    func=compute_technical_indicators,
    name="compute_technical_indicators",
    description=(
        "Computes technical analysis indicators for a stock. "
        "Available indicators: sma_20, ema_20, rsi_14, macd, bbands, trend. "
        "Returns current values and recent history for each indicator."
    ),
    args_schema=ComputeIndicatorsInput
)

generate_chart_tool = StructuredTool.from_function(
    func=generate_stock_chart,
    name="generate_stock_chart",
    description= (
        "Generates a price or RSI chart for a stock. "
        "Can overlay technical indicators on price charts. "
        "Returns path to saved chart image."
    ),
    args_schema=GenerateChartInput
)

# List of all analysis tools
ANALYSIS_TOOLS = [compute_indicators_tool, generate_chart_tool]