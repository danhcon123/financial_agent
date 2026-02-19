"""
Chart generation for financial data visualization.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import date as date_type
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

class ChartGenerator:
    """Generate financial charts using matplotlib"""
    def __init__(self, output_dir: str = "data/processed/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def generate_price_chart(
        self,
        dates: List[date_type],
        prices: List[float],
        ticker: str,
        indicators: Optional[Dict[str, List[float]]] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Generate price chart with optional indicators.
        
        Args:
            dates: List of dates
            prices: List of closing prices
            ticker: Stock ticker symbol
            indicators: Dict of indicator_name -> values (e.g., {'SMA_20': [...]})
            filename: Custom filename (auto-generated if None)
        
        Returns:
            Path to saved chart
        """
        if len(dates) != len(prices):
            raise ValueError(f"Dates ({len(dates)}) and prices ({len(prices)}) must be same length")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot price
        ax.plot(dates, prices, label=f'{ticker} Close', linewidth=2, color='#2E86AB')
        
        # Plot indicators
        if indicators:
            colors = ['#A23B72', '#F18F01', '#C73E1D', '#6A994E']
            for i, (name, values) in enumerate(indicators.items()):
                if len(values) == len(dates):
                    color = colors[i % len(colors)]
                    ax.plot(dates, values, label=name, linewidth=1.5, 
                           linestyle='--', alpha=0.8, color=color)
        
        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_title(f'{ticker} Price Chart', fontsize=14, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save
        if not filename:
            filename = f"{ticker}_price_{dates[-1]}.png"
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Chart saved: {filepath}")
        return str(filepath)
    
    def generate_rsi_chart(
        self,
        dates: List[date_type],
        rsi_values: List[float],
        ticker: str,
        filename: Optional[str] = None
    ) -> str:
        """Generate RSI indicator chart"""
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Plot RSI
        ax.plot(dates, rsi_values, label='RSI', linewidth=2, color='#8338EC')
        
        # Add overbought/oversold lines
        ax.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
        ax.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
        ax.axhline(y=50, color='gray', linestyle=':', alpha=0.3)
        
        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('RSI', fontsize=12)
        ax.set_title(f'{ticker} Relative Strength Index (RSI)', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if not filename:
            filename = f"{ticker}_rsi_{dates[-1]}.png"
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"RSI chart saved: {filepath}")
        return str(filepath)