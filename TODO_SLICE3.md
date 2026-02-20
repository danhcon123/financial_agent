# Slice 3: Fundamental Data & Enhanced Analytics

## Priority 1: Automatic Sector-Based Benchmark Selection

### Current State (Slice 2)
- ✅ Hardcoded: Tech stocks → QQQ, Everything else → SPY
- ✅ Works for 80% of common use cases
- ❌ Doesn't scale to all stocks/sectors

### Goal
Automatically detect stock sector and select appropriate benchmark ETF.

### Implementation Plan

#### Step 1: Add Sector Detection (yfinance)
**File:** `src/sandbox/analytics.py`
```python
def get_sector_from_yfinance(ticker: str) -> Optional[str]:
    """
    Get sector for a stock using yfinance.
    
    Returns:
        Sector name (e.g., 'Technology', 'Energy')
    """
    import yfinance as yf
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get('sector')
        
        # Cache result to avoid repeated API calls
        # TODO: Add caching mechanism (redis or local file)
        
        return sector
    except Exception as e:
        logger.warning(f"Could not get sector for {ticker}: {e}")
        return None
```

#### Step 2: Sector to ETF Mapping
**File:** `src/sandbox/analytics.py`
```python
def get_benchmark_from_sector(sector: Optional[str]) -> str:
    """
    Map sector to appropriate benchmark ETF.
    
    Sector ETF Mapping:
    - Technology → XLK (Tech Select Sector SPDR)
    - Financial Services → XLF (Financial Select Sector SPDR)
    - Healthcare → XLV (Health Care Select Sector SPDR)
    - Energy → XLE (Energy Select Sector SPDR)
    - Consumer Cyclical → XLY (Consumer Discretionary SPDR)
    - Consumer Defensive → XLP (Consumer Staples SPDR)
    - Industrials → XLI (Industrial Select Sector SPDR)
    - Real Estate → XLRE (Real Estate Select Sector SPDR)
    - Utilities → XLU (Utilities Select Sector SPDR)
    - Basic Materials → XLB (Materials Select Sector SPDR)
    - Communication Services → XLC (Communication Services SPDR)
    
    Args:
        sector: Sector name from yfinance
    
    Returns:
        ETF ticker to use as benchmark
    """
    if not sector:
        return 'SPY'  # Default to S&P 500
    
    sector_map = {
        'Technology': 'XLK',
        'Financial Services': 'XLF',
        'Healthcare': 'XLV',
        'Energy': 'XLE',
        'Consumer Cyclical': 'XLY',
        'Consumer Defensive': 'XLP',
        'Industrials': 'XLI',
        'Real Estate': 'XLRE',
        'Utilities': 'XLU',
        'Basic Materials': 'XLB',
        'Communication Services': 'XLC',
    }
    
    return sector_map.get(sector, 'SPY')
```

#### Step 3: Update `get_appropriate_benchmark()`
**File:** `src/sandbox/analytics.py`

Replace hardcoded logic with automatic detection:
```python
def get_appropriate_benchmark(ticker: str) -> str:
    """
    Intelligently select benchmark for a ticker.
    
    1. Try to get sector from yfinance
    2. Map sector to sector ETF
    3. Fall back to SPY if unavailable
    """
    sector = get_sector_from_yfinance(ticker)
    return get_benchmark_from_sector(sector)
```

#### Step 4: Add Caching (Performance Optimization)
**Why:** Avoid hitting yfinance API every time

**Options:**
- Simple: JSON file cache (`data/cache/sectors.json`)
- Better: Redis cache (requires setup)
- Production: Database table

**Simple Implementation:**
```python
import json
from pathlib import Path

SECTOR_CACHE_FILE = Path("data/cache/sectors.json")

def load_sector_cache() -> Dict[str, str]:
    if SECTOR_CACHE_FILE.exists():
        with open(SECTOR_CACHE_FILE) as f:
            return json.load(f)
    return {}

def save_sector_cache(cache: Dict[str, str]):
    SECTOR_CACHE_FILE.parent.mkdir(exist_ok=True)
    with open(SECTOR_CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def get_sector_from_yfinance(ticker: str) -> Optional[str]:
    # Check cache first
    cache = load_sector_cache()
    if ticker in cache:
        return cache[ticker]
    
    # Fetch from yfinance
    try:
        stock = yf.Ticker(ticker)
        sector = stock.info.get('sector')
        
        # Save to cache
        cache[ticker] = sector
        save_sector_cache(cache)
        
        return sector
    except Exception as e:
        logger.warning(f"Could not get sector for {ticker}: {e}")
        return None
```

#### Step 5: Testing
**File:** `tests/test_slice3.py`
```python
def test_sector_detection():
    """Test automatic sector detection"""
    sector = get_sector_from_yfinance("AAPL")
    assert sector == "Technology"
    
    benchmark = get_benchmark_from_sector(sector)
    assert benchmark == "XLK"

def test_sector_cache():
    """Test that sector is cached after first fetch"""
    # First call - fetches from yfinance
    sector1 = get_sector_from_yfinance("MSFT")
    
    # Second call - uses cache (should be instant)
    import time
    start = time.time()
    sector2 = get_sector_from_yfinance("MSFT")
    elapsed = time.time() - start
    
    assert sector1 == sector2
    assert elapsed < 0.01  # Should be nearly instant from cache
```

### Expected Output After Implementation

**Before (Slice 2):**
```
vs SPY: +2.1% (outperforming)
```

**After (Slice 3):**
```
vs XLK: +2.1% (outperforming Tech sector)
vs XLF: -1.5% (underperforming Financial sector)
vs XLE: +8.2% (strong outperformance vs Energy sector)
```

### Dependencies
- `yfinance` (already installed)
- No additional packages needed

### Time Estimate
- Step 1-3: ~30 minutes
- Step 4 (caching): ~20 minutes
- Step 5 (testing): ~15 minutes
- **Total: ~1 hour**

---

## Priority 2: Fundamental Data Integration

### Data Sources to Add
1. **Earnings Data** (yfinance)
   - EPS (earnings per share)
   - Revenue
   - Profit margins
   - Growth rates

2. **Valuation Metrics** (yfinance)
   - P/E ratio
   - P/B ratio
   - PEG ratio
   - Compare to sector averages

3. **Company Info** (yfinance)
   - Market cap
   - Industry
   - Employee count
   - Description

### Implementation
**File:** `src/data/fetchers/fundamental_data.py` (new)
```python
def fetch_fundamentals(ticker: str) -> Dict[str, Any]:
    """
    Fetch fundamental data using yfinance.
    
    Returns:
        Dict with earnings, valuation, company info
    """
    import yfinance as yf
    
    stock = yf.Ticker(ticker)
    info = stock.info
    
    return {
        'earnings': {
            'eps': info.get('trailingEps'),
            'forward_eps': info.get('forwardEps'),
            'revenue': info.get('totalRevenue'),
            'profit_margin': info.get('profitMargins'),
        },
        'valuation': {
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'peg_ratio': info.get('pegRatio'),
            'price_to_book': info.get('priceToBook'),
        },
        'company': {
            'market_cap': info.get('marketCap'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'employees': info.get('fullTimeEmployees'),
        }
    }
```

### Time Estimate: ~2 hours

---

## Priority 3: News Sentiment Analysis

### Data Sources
1. **News API** or **Alpha Vantage**
2. **Sentiment Analysis** (VADER or FinBERT)

### Implementation Plan
- Fetch recent news headlines
- Analyze sentiment (positive/negative/neutral)
- Add to evidence with confidence score

### Time Estimate: ~3 hours

---

## Priority 4: SEC Filings Integration

### Data Sources
- SEC Edgar API
- Extract key metrics from 10-K, 10-Q

### Time Estimate: ~4 hours

---

## Total Slice 3 Estimate: ~10-12 hours