"""
Data quality validation for OHLCV price data.
"""

from typing import List
from datetime import timedelta

from src.models.market_data import OHLCVRow, ValidationResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

class OHLCVValidator:
    """Validates OHLCV data for quality and consistency"""
    
    def __init__(
        self,
        max_gap_days: int = 5,
        max_price_change_pct: float = 50.0,
        min_volume: int = 0
    ):
        """
        Initialize validator with thresholds.

        Args: 
            max_gap_days: Maximum allowed gap between trading days
            max_price_change_pct: Maximum % price change between consecutive days
            min_volume: Minimum acceptable volume (0 = no minimum) 
        """
        self.max_gap_days = max_gap_days
        self.max_price_change_pct = max_price_change_pct
        self.min_volume = min_volume

    def validate(self, rows: List[OHLCVRow]) -> ValidationResult:
        """
        Validate OHLCV data quality
        
        Checks:
        - Date gaps (missing trading days)
        - Price outliers (extreme % changes)
        - Volume anomalies (zero volume, extreme spikes)
        - OHLC consistency (already validated by Pydantic)

        Returns:
            ValidationResult with errors and warning
        """
        result = ValidationResult(is_valid=True, rows_checked=len(rows))

        if not rows:
            result.add_error("No data validate")
            return result
        
        # Sort by date
        sorted_rows = sorted(rows, key=lambda x: x.date)
        
        # Check for duplicates
        dates_seen = set()
        for row in sorted_rows:
            if row.date in dates_seen:
                result.add_error(f"Duplicate date found: {row.date}")
            dates_seen.add(row.date)
        
        # Validate each row and check consistency
        prev_row = None
        for i, row in enumerate(sorted_rows):
            # Individual row validation (Pydantic already checks OHLC consistency)
            if row.volume < self.min_volume:
                result.add_warning(
                    f"Low volume on {row.date}: {row.volume} shares"
                )

            # Check consecutive days
            if prev_row:
                self._check_date_gap(prev_row, row, result)
                self._check_price_jump(prev_row, row, result)

            prev_row = row
        
        # result.rows_passed = len(sorted_rows) - len(result.errors)

        if result.is_valid:
            logger.info(
                f"Validation passed: {result.rows_passed}/{result.rows_checked} rows "
            )
        else:
            logger.error(
                f"Validation failed: {len(result.errors)} errors, "
                f"{len(result.warnings)} warnings"
            )

        return result
    
    def _check_date_gap(
        self,
        prev: OHLCVRow,
        current: OHLCVRow,
        result: ValidationResult
    ) -> None:
        """Check for excessive gaps between trading days"""
        gap = (current.date - prev.date).days

        # Allow for weekends (2 days) + holidays (up to max_gap_days)
        if gap > self.max_gap_days:
            result.add_warning(
                f"Large date gap: {gap} days between {prev.date} and {current.date}"
            )

    def _check_price_jump(
        self,
        prev: OHLCVRow,
        current: OHLCVRow,
        result: ValidationResult
    ) -> None:
        """Check for extreme price changes between consecutive days"""
        pct_change = abs((current.close - prev.close) / prev.close) * 100

        if pct_change > self.max_price_change_pct:
            result.add_error(
                f"Extreme price change on {current.date}: "
                f"{pct_change:.1f}% from ${prev.close:.2f} to ${current.close:.2f}"
            )