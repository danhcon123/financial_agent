"""
Market data schemas for price information
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import date, datetime

class OHLCVRow(BaseModel):
    """Single day of OHLCV (Open, High, Low, Close, Volume) data"""

    ticker: str = Field(..., description="Stock ticker symbol")
    date: date = Field(..., description="Trading date")
    open: float = Field(..., description="Opening price", gt=0)
    high: float = Field(..., description="Highest price", gt=0)
    low: float = Field(..., description="Lowest price", gt=0)
    close: float = Field(..., description="Closing price", gt=0)
    volume: int = Field(..., description="Trading volume", ge=0)

    @field_validator('high')
    @classmethod
    def high_gte_low(cls, v, info):
        """Validate high >= low"""
        if 'low' in info.data and v < info.data['low']:
            raise ValueError(f"High ({v}) must be >= Low ({info.data['low']})")
        return v
    
    @field_validator('high')
    @classmethod
    def high_gte_open_close(cls, v, info):
        """Validate high >= open and close"""
        if 'open' in info.data and v < info.data['open']:
            raise ValueError(f"High ({v}) must be >= Open ({info.data['open']})")
        if 'close' in info.data and v < info.data['close']:
            raise ValueError(f"High ({v}) must be >= Close ({info.data['close']})")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "date": "2026-02-13",
                "open": 189.50,
                "high": 191.25,
                "low": 188.75,
                "close": 190.80,
                "volume": 52340000
            }
        }
    
class PriceDataSummary(BaseModel):
    """Summary statistics for price data"""

    ticket: str
    start_date: date
    end_date: date
    row_count: int
    latest_close: float
    price_change_pct: float = Field(..., subscription="% change from start to end")
    avg_volume: float
    avg_price: float
    min_price: float
    max_price: float

    def to_evidence_claim(self) -> str:
        """Format as evidence claim for analyst"""
        direction = "up" if self.price_change_pct > 0 else "down"
        return(
            f"{self.ticker} closed at ${self.latest_close:.2f} on {self.end_date}, "
            f"{direction} {abs(self.price_change_pct):.1f}% over {self.row_count} trading days."
            f"Average volume: {self.avg_volume:, .0f} shares"
            f"Price range: ${self.min_price:.2f} - ${self.max_price:.2f}."
        )
    
class ValidationResult(BaseModel):
    """Result of data validation"""
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rows_checked: int = 0
    row_passed: int = 0
    
    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)