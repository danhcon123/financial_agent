"""
Yahoo Finance data fetcher using yfinance library.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional

from src.models.market_data import OHLCVRow, PriceDataSummary
from src.utils.logger import get_logger

logger = get_logger(__name__)

class YahooFinanceFetcher:
    """Fetches historical stock price data from Yahoo Finance"""

    def __init__(self):
        self.source = "yahoo_finance"

    def fetch(
            self,
            ticker: str,
            days: int = 252, # 1 trading year
            end_date: Optional[datetime] = None
    ) -> tuple[List[OHLCVRow], PriceDataSummary]:
        """
        Fetch historical OHLCV data for a ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            days: Number of calendar days of history
            end_date: End date (default: today)

        Returns:
            Tuples of (list of OHLCV rows, summary statistics)

        Raises:
            ValueError: If ticker is invalid or no data returned
        """
        ticker = ticker.upper().strip()
        end_date = end_date or datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Fetching {ticker} data from {start_date.date()} to {end_date.date()}")

        try:
            # Fetch from yfinance
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, auto_adjust=False)

            if df.empty:
                raise ValueError(f"No data returned for ticker {ticker}")
            
            logger.info(f"Retrieved {len(df)} rows for {ticker}")

            # Convert to our schema
            rows = self._convert_to_ohlcv(df, ticker)

            # Generate summary
            summary = self._generate_summary(rows, ticker)
            
            logger.info(
                f"{ticker} summary: {summary.row_count} days, "
                f"latest close ${summary.latest_close:.2f}, "
                f"change {summary.price_change_pct:+.2f}%"
            )

            return rows, summary
        
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {e}")
            raise ValueError(f"Failed to fetch {ticker} from Yahoo Finance: {e}") from e

    def _convert_to_ohlcv(self, df: pd.DataFrame, ticker: str) -> List[OHLCVRow]:
        """Convert pandas DataFrame to list of OHLCVRow objects"""
        rows = []
        
        for date_idx, row in df.iterrows():
            try:
                ohlcv = OHLCVRow(
                    ticker=ticker,
                    date=date_idx.date(),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume'])
                )
                rows.append(ohlcv)

            except Exception as e:
                logger.warning(f"Skipping row {date_idx} due to validation error: {e}")
                continue
        return rows
    
    def _generate_summary(self, rows: List[OHLCVRow], ticker: str) -> PriceDataSummary:
        """Generate summary statistics from OHLCV data"""
        if not rows:
            raise ValueError("Cannot generate summary from empty data")
        
        # Sort by date to ensure correct ordering
        rows = sorted(rows, key=lambda x: x.date)
        
        closes = [r.close for r in rows]
        volumes = [r.volume for r in rows]
        all_prices = []
        for r in rows:
            all_prices.extend([r.open, r.high, r.low, r.close])

        first_close = rows[0].close
        last_close = rows[-1].close
        price_change_pct = ((last_close - first_close) / first_close) * 100

        return PriceDataSummary(
            ticker=ticker,
            start_date=rows[0].date,
            end_date=rows[-1].date,
            row_count=len(rows),
            latest_close=last_close,
            price_change_pct=price_change_pct,
            avg_volume=sum(volumes) / len(volumes),
            avg_price=sum(closes) / len(closes),
            min_price=min(all_prices),
            max_price=max(all_prices)
        )
    
# Convention function for CLI testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.data.fetchers.yahoo_finance TICKER [DAYS]")
        sys.exit(1)

    ticker = sys.argv[1]
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 90

    fetcher = YahooFinanceFetcher()
    rows, summary = fetcher.fetch(ticker, days)

    print(f"\n{'='*60}")
    print(f"Fetched {summary.row_count} days of data for {ticker}")
    print(f"{'='*60}")
    print(f"Period: {summary.start_date} to {summary.end_date}")
    print(f"Price Change: {summary.price_change_pct:+2f}%")
    print(f"Avg Volume: ${summary.avg_volume:,.0f}")
    print(f"Price Range: ${summary.min_price:.2f} - ${summary.max_price:.2f}")
    print(f"\nEvidence Claim:\n{summary.to_evidence_claim()}")
    print(f"{'='*60}\n")