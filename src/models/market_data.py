"""
Market data schemas for price information
"""

from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Optional
from datetime import date, datetime
from decimal import Decimal

class OHLCVRow(BaseModel):
    """Single day of OHLCV (Open, High, Low, Close, Volume) data"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ticker":"AAPL",
                "date":"2026-02-13",
                "open": 189.50,
                "high": 191.25,
                "low": 188.75,
                "close": 190.80,
                "volume": 52340000
            }
        }
    )

    ticker: str = Field(..., description="Stock ticker symbol")
    date: datetime = Field(..., description="Trading date")
    open: float = Field(..., description="Opening price", gt=0)
    high: float = Field(..., description="Highest price", gt=0)
    low: float = Field(..., description="Lowest price", gt=0)
    close: float = Field(..., description="Closing price", gt=0)
    volume: int = Field(..., description="Trading volume", ge=0)

    @model_validator(mode='after')
    def validate_ohlc_consistency(self) -> 'OHLCVRow':
        """
        Validate OHLC relationships:
        - High must be the highest price of the day (>= open, low, close)
        - Low must be the lowest price of the day (<= open, high, close)
        """
        # High must be >= all other prices
        if self.high < self.low:
            raise ValueError(f"High ({self.high}) must be >= Low ({self.low})")
        if self.high < self.open:
            raise ValueError(f"High ({self.high}) must be >= Open ({self.open})")
        if self.high < self.close:
            raise ValueError(f"High ({self.high}) must be >= Close ({self.close})")
        
        # Low must be <= all other prices
        if self.low > self.open:
            raise ValueError(f"Low ({self.low}) must be <= Open ({self.open})")
        if self.low > self.close:
            raise ValueError(f"Low ({self.low}) must be <= Close ({self.close})")
        
        return self
    
class PriceDataSummary(BaseModel):
    """Summary statistics for price data"""

    ticker: str
    start_date: date
    end_date: date
    row_count: int
    latest_close: float
    price_change_pct: float = Field(..., description="% change from start to end")
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
            f"Average volume: {self.avg_volume:,} shares."
            f"Price range: ${self.min_price:.2f} - ${self.max_price:.2f}."
        )
    
class ValidationResult(BaseModel):
    """Result of data validation"""
    model_config = ConfigDict(validate_assignment=True)

    is_valid: bool = True
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rows_checked: int = 0
    
    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    @property
    def rows_passed(self) -> int:
        """Calculate rows passed from checked - errors"""
        return self.rows_checked - len(self.errors)