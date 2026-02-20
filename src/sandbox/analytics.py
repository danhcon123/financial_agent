"""
Technical analysis functions for financial data
Safe, pure functions - no file I/O, no network access
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)

def compute_sma(prices: List[float], period: int = 20) -> List[float]:
    """Compute Simple Moving Average"""
    if len(prices) < period:
        return [np.nan] * len(prices)
    
    df = pd.DataFrame({'price': prices})
    sma = df['price'].rolling(window=period).mean()
    return sma.tolist()

def compute_ema(prices: List[float], period: int=20) -> List[float]:
    """Compute Exponential Moving Average"""
    if len(prices) < period:
        return [np.nan] * len(prices)
    
    df = pd.DataFrame({'price': prices})
    ema = df['price'].ewm(span=period, adjust=False).mean()
    return ema.tolist()

def compute_rsi(prices: List[float], period: int = 14) -> List[float]:
    """Compute Relative Strength Index"""
    if len(prices) < period:
        return [np.nan] * len(prices)
    
    df = pd.DataFrame({'price': prices})
    delta = df['price'].diff()
    
    gain = (delta.where(delta > 0,0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0,0)).rolling(window=period).mean()
    
    rs = gain/loss
    rsi = 100 - (100/(1+rs))
    return rsi.tolist()

def compute_macd(
    prices: List [float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Dict[str, List[float]]:
    """Compute MACD (Moving Average Convergence Divergence)"""
    if len(prices) < slow_period:
        nan_list = [np.nan] * len(prices)
        return{
            'macd': nan_list,
            'signal': nan_list,
            'histogram': nan_list
        }
    
    df = pd.DataFrame({'price': prices})
    
    ema_fast = df['price'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['price'].ewm(span=slow_period, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return {
        'macd': macd_line.tolist(),
        'signal': signal_line.tolist(),
        'histogram': histogram.tolist()
    }

def compute_bollinger_bands(
    prices: List[float],
    period: int = 20,
    num_std: float = 2.0
) -> Dict[str, List[float]]:
    """Compute Bollinger Bands"""
    if len(prices) < period:
        nan_list = [np.nan] * len(prices)
        return {
            'upper': nan_list,
            'middle': nan_list,
            'lower': nan_list
        }
    
    df = pd.DataFrame({'price': prices})
    
    middle = df['price'].rolling(window=period).mean()
    std = df['price'].rolling(window=period).std()

    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    return{
        'upper': upper.tolist(),
        'middle': middle.tolist(),
        'lower': lower.tolist()
    }

def analyze_trend(prices: List[float], sma_period: int = 20) -> Dict[str, Any]:
    """
    Analyze price trend relative to SMA.

    Returns:
        Dict with trend analysis including:
        - current_price
        - sma_value
        - percent_from_sma
        - trend (above/below)
        - signal (bullish/bearish/neutral)
    """
    if len(prices) < sma_period + 1:
        return {
            'error': f'Insufficient data: need {sma_period + 1} prices, got {len(prices)}'
        }
    
    current_price = prices[-1]
    sma_values = compute_sma(prices, sma_period)
    sma_value = sma_values[-1]

    if np.isnan(sma_value):
        return {'error': 'Could not compute SMA'}
    
    percent_from_sma = ((current_price - sma_value) / sma_value) * 100

    # Determin signal
    if percent_from_sma > 2:
        signal = 'bullish'
    elif percent_from_sma < -2:
        signal = 'bearish'
    else:
        signal = 'neutral'

    return {
        'current_price': round(current_price, 2),
        'sma_value': round(sma_value, 2),
        'percent_from_sma': round(percent_from_sma, 2),
        'trend': 'above' if current_price > sma_value else 'below',
        'signal': signal
    }

def analyze_volume(
        volumes: List[float],
        period: int = 20
) -> Dict [str, Any]:
    """
    Analyze volume relative to average
    Args:
        volumes: List of volume values
        period: Period for average calculation (default 20)

    Returns:
        Dict with volume analysis including
        - current_volume
        - avg_volume
        - volume_ratio (current / avg)
        - signal (high/low/normal)
    """
    if len(volumes) < period:
        return {'error': f'Need at least {period + 1} volume data points'}
    
    current_volume = volumes[-1]
    avg_volume = sum(volumes[-period-1:-1]) / period
    if avg_volume == 0:
        return {'error': 'Average volume is zero'}
    
    volume_ratio = current_volume / avg_volume

    # Determine signal
    if volume_ratio > 2.0:
        signal = 'very_high'
        interpretation = 'Strong institutional interest'
    elif volume_ratio > 1.5:
        signal = 'high'
        interpretation = 'Above average activity'
    elif volume_ratio < 0.5:
        signal = 'low'
        interpretation = 'Below average activity, caution'
    else:
        signal = 'normal'
        interpretation = 'Normal trading activity'
    
    return {
        'current_volume': int(current_volume),
        'avg_volume': int(avg_volume),
        'volume_ratio': round(volume_ratio, 2),
        'signal': signal,
        'interpretation': interpretation
    }

def get_appropriate_benchmark(ticker: str) -> str:
    """
    Select appropriate benchmark for comparison.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Benchmark ETF ticker (QQQ, SPY, etc.)
    """
    ticker = ticker.upper()
    
    # Tech/Growth stocks → Use QQQ (Nasdaq 100)
    NASDAQ_STOCKS = [
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'AMZN', 'NVDA', 'TSLA',
        'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'CSCO', 'AVGO', 'TXN',
        'QCOM', 'NFLX', 'PYPL', 'COST', 'SBUX', 'AMAT', 'MU', 'LRCX'
    ]
    if ticker in NASDAQ_STOCKS:
        return 'QQQ'
    
    # TODO (Slice 3): Add yfinance sector lookup for automatic detection
    # For now, default to S&P 500 for all other stocks
    return 'SPY'


def compare_to_benchmark(
    ticker_returns: float,
    ticker: str,
    period_days: int = 60
) -> Dict[str, Any]:
    """
    Compare stock performance to appropriate benchmark.
    
    Automatically selects QQQ (Nasdaq) or SPY (S&P 500) based on ticker.
    
    Args:
        ticker_returns: Returns for the stock (%)
        ticker: Stock ticker symbol
        period_days: Period for comparison
    
    Returns:
        Dict with comparative analysis
    """
    from src.data.storage import DuckDBStorage
    from src.data.fetchers.yahoo_finance import YahooFinanceFetcher
    
    # ✅ Select appropriate benchmark
    benchmark_ticker = get_appropriate_benchmark(ticker)
    
    try:
        storage = DuckDBStorage()
        
        # Try to fetch benchmark from database
        bench_rows = storage.fetch(benchmark_ticker, limit=period_days + 10)
        
        # If no benchmark data, fetch it
        if not bench_rows or len(bench_rows) < period_days:
            logger.info(f"Fetching {benchmark_ticker} data for comparison...")
            fetcher = YahooFinanceFetcher()
            bench_data, _ = fetcher.fetch(benchmark_ticker, days=period_days)
            if bench_data:
                storage.store(bench_data, replace=True)
                bench_rows = storage.fetch(benchmark_ticker, limit=period_days + 10)
        
        storage.close()
        
        if not bench_rows or len(bench_rows) < 2:
            return {
                'error': f'Insufficient {benchmark_ticker} data for comparison',
                'ticker_returns': round(ticker_returns, 2),
                'benchmark': benchmark_ticker
            }
        
        # Calculate benchmark returns
        bench_rows = sorted(bench_rows, key=lambda x: x.date)[-period_days:]
        bench_start = bench_rows[0].close
        bench_end = bench_rows[-1].close
        bench_returns = ((bench_end - bench_start) / bench_start) * 100
        
        # Calculate relative strength
        relative_strength = ticker_returns - bench_returns
        
        # Determine signal
        if relative_strength > 5:
            signal = 'strong_outperformance'
            interpretation = f'{ticker} significantly outperforming {benchmark_ticker}'
        elif relative_strength > 2:
            signal = 'outperforming'
            interpretation = f'{ticker} outperforming {benchmark_ticker}'
        elif relative_strength < -5:
            signal = 'strong_underperformance'
            interpretation = f'{ticker} significantly underperforming {benchmark_ticker}'
        elif relative_strength < -2:
            signal = 'underperforming'
            interpretation = f'{ticker} underperforming {benchmark_ticker}'
        else:
            signal = 'inline'
            interpretation = f'{ticker} moving in line with {benchmark_ticker}'
        
        return {
            'ticker_returns': round(ticker_returns, 2),
            'benchmark': benchmark_ticker,
            'benchmark_returns': round(bench_returns, 2),
            'relative_strength': round(relative_strength, 2),
            'signal': signal,
            'interpretation': interpretation
        }
    
    except Exception as e:
        logger.error(f"Error in benchmark comparison: {e}")
        return {
            'error': str(e),
            'ticker_returns': round(ticker_returns, 2),
            'benchmark': benchmark_ticker
        }