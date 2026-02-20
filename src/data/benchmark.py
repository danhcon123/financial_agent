"""
Sector-aware benchmark selector for financial analysis

Detects a stock's GICS sector via yfinance and maps it to
the appropriate Select Sector SPDR ETF. Falls back to SPY
if sector is unknown or detection fails.

Usage:
    selector = BenchmarkSelector()
    info = selector.get_benchmark("NVDA")
    print(info.etf_ticker) # "XLK"
    print(info.sector) # "Technology"
"""

import yfinance as yf
from pydantic import BaseModel, Field
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# GICS Sector → SPDR ETF Mapping (stable, hardcoded intentionally)
# Source: https://www.ssga.com/us/en/intermediary/etfs/fund-finder
# ============================================================================
SECTOR_TO_ETF: dict[str, str] = {
    "Technology":             "XLK",
    "Information Technology": "XLK",  # yfinance uses this variant
    "Financial Services":     "XLF",
    "Financials":             "XLF",
    "Healthcare":             "XLV",
    "Health Care":            "XLV",  # yfinance uses this variant
    "Energy":                 "XLE",
    "Consumer Cyclical":      "XLY",  # yfinance name for Consumer Discretionary
    "Consumer Discretionary": "XLY",
    "Consumer Defensive":     "XLP",  # yfinance name for Consumer Staples
    "Consumer Staples":       "XLP",
    "Industrials":            "XLI",
    "Utilities":              "XLU",
    "Basic Materials":        "XLB",  # yfinance name for Materials
    "Materials":              "XLB",
    "Real Estate":            "XLRE",
    "Communication Services": "XLC",
}

# Sector average P/E ratios (approximate, updated periodically)
# Source: Damodaran NYU dataset - these are rough long-term averages
# TODO (Slice 3 Phase 2): Replace with live fetching from a data source
SECTOR_PE_AVERAGES: dict[str, float] = {
    "Technology":             28.5,
    "Information Technology": 28.5,
    "Financial Services":     14.2,
    "Financials":             14.2,
    "Healthcare":             22.1,
    "Health Care":            22.1,
    "Energy":                 12.8,
    "Consumer Cyclical":      24.3,
    "Consumer Discretionary": 24.3,
    "Consumer Defensive":     21.5,
    "Consumer Staples":       21.5,
    "Industrials":            20.7,
    "Utilities":              18.9,
    "Basic Materials":        16.4,
    "Materials":              16.4,
    "Real Estate":            35.2,  # REITs have elevated P/E by nature
    "Communication Services": 19.8,
}

SECTOR_PB_AVERAGES: dict[str, float] = {
    "Technology": 6.8,
    "Information Technology": 6.8,
    "Financial Services": 1.4,
    "Financials": 1.4,
    "Healthcare": 4.2,
    "Health Care": 4.2,
    "Energy": 1.8,
    "Consumer Cyclical": 4.9,
    "Consumer Discretionary": 4.9,
    "Consumer Defensive": 5.1,
    "Consumer Staples": 5.1,
    "Industrials": 3.8,
    "Utilities":              1.9,
    "Basic Materials":        2.3,
    "Materials":              2.3,
    "Real Estate":            2.1,
    "Communication Services": 3.4,
}

FALLBACK_BENCHMARK = "SPY"
FALLBACK_SECTOR = "Unknown"

# ============================================================================
# Return Object
# ============================================================================

class BenchmarkInfo(BaseModel):
    """
    Rich benchmark metadata for a stock, used in both technical and fundamental analysis.
    """
    etf_ticker: str = Field(..., description="Benchmark ETF to compare against (e.g. XLK)")
    sector: str = Field(..., description="GICS sector name as returned by yfinance")
    industry: str = Field("Unknown", description="More specific industry classification")
    sector_pe_avg: Optional[float] = Field(None, description="Approximate sector average P/E ratio")
    sector_pb_avg: Optional[float] = Field(None, description="Approximate sector average P/B ratio")
    is_fallback: bool = Field(False, description="True if we fell back to SPY due to detection failure")
    notes: str = Field("", description="Human-readable explanation of how benchmark was selected")

    def to_evidence_claim(self) -> str:
        """Format benchmark info as an evidence string for the analyst agent."""
        if self.is_fallback:
            return(
                f"Benchmark: {self.etf_ticker} (S&P500, used as fallback - sector detection failed). "
                f"Note: {self.notes}"
            )
        
        parts = [
            f"Sector: {self.sector}",
            f"Industry: {self.industry}",
            f"Benchmark ETF: {self.etf_ticker}",
        ]

        if self.sector_pe_avg:
            parts.append(f"Sector avg P/E: {self.sector_pe_avg:.1f}")
        if self.sector_pb_avg:
            parts.append(f"Sector avg P/B: {self.sector_pb_avg:.1f}")

        return " | ".join(parts)

# ============================================================================
# BenchmarkSelector
# ============================================================================

class BenchmarkSelector:
    """
    Detects a stock's sector via yfinance and returns a BenchmarkInfo object
    containing the appropriate SPDR ETF and sector metadata
    """

    def get_benchmark(self, ticker: str) -> BenchmarkInfo:
        """
        Main entry point. Returns BenchmarkInfo for the given ticker.

        Args:
            ticker: US stock ticker symbol (e.g. 'NVDA', 'JPM')

        Returns:
            BenchmarkInfo with ETF, sector, industry, and avg valuation ratios
            Always returns a valid object - never raises. Falls back to SPY on error
        """
        ticker = ticker.upper().strip()
        logger.info(f"Detecting sector for {ticker}")
        try:
            sector, industry = self._fetch_sector(ticker)
        except Exception as e:
            logger.warning(f"Sector detection failed for {ticker}: {e}")
            return self._make_fallback(ticker, reason=f"yfinance error: {e}")
        
        if not sector or sector == "N/A":
            return self._make_fallback(ticker, reason=f"yfinance returned empty sector")
        
        etf = SECTOR_TO_ETF.get(sector)
        if not etf:
            logger.warning(f"No ETF mapping for sector '{sector}' (ticker: {ticker}), falling back to SPY")
            return self._make_fallback(
                ticker,
                reason=f"No ETF mapping for sector '{sector}'"
            )
        
        sector_pe = SECTOR_PE_AVERAGES.get(sector)
        sector_pb = SECTOR_PB_AVERAGES.get(sector)

        logger.info(f"{ticker} -> sector = '{sector}', industry='{industry}', ETF={etf}")

        return BenchmarkInfo(
            etf_ticker=etf,
            sector=sector,
            industry=industry or "Unknown",
            sector_pe_avg=sector_pe,
            sector_pb_avg=sector_pb,
            is_fallback=False,
            notes=f"Sector detected via yfinance for {ticker}"
        )
    
    def _fetch_sector(self, ticker: str) -> tuple[str, str]:
        """
        Fetch sector and industry from yfinance
        
        Returns:
            (sector, industry) strings. May be empty or 'N/A'.
        """
        stock = yf.Ticker(ticker)
        info = stock.info # single network call, returns a dict

        sector = info.get("sector", "") or ""
        industry = info.get("industry", "") or ""
        return sector, industry
    

    def _make_fallback(self, ticker: str, reason: str) -> BenchmarkInfo:
        """Construct a SPY fallback BenchmarkInfo with explanation."""
        return BenchmarkInfo(
            etf_ticker=FALLBACK_BENCHMARK,
            sector=FALLBACK_SECTOR,
            industry="Unknown",
            sector_pe_avg=None,
            sector_pb_avg=None,
            is_fallback=True,
            notes=f"Fallback to SPY for {ticker}. Reason {reason}"
        )