"""
DuckDB storage layer for market data
"""
import hashlib
import json
import duckdb
from typing import List, Optional
from pathlib import Path
from datetime import date, datetime

from src.models.market_data import OHLCVRow
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DuckDBStorage:
    """DuckDB-based storage for OHLCV price data"""
    
    def __init__(self, db_path: str = "data/market_data.db"):
        """
        Initialize DuckDB connection and create schema.

        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = db_path

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self.conn = duckdb.connect(db_path)
        
        # Create schema
        self._create_schema()
        
        logger.info(f"Connected to DuckDB: {db_path}")

    def _create_schema(self) -> None:
        """Create tables if they don't exist to store price data, news and volume coverage (gdelt_coverage)"""
        # Price data table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS prices(
                ticker VARCHAR NOT NULL,
                date DATE NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                volume BIGINT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, date)
            )
        """)
        # News articles table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS news_articles(
                id VARCHAR PRIMARY KEY,
                ticker VARCHAR NOT NULL,
                title VARCHAR,
                url VARCHAR,
                source VARCHAR,
                published_at TIMESTAMP,
                sentiment_score DOUBLE,
                sentiment_label VARCHAR,
                topics          VARCHAR,
                fetched_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                summary VARCHAR
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS gdelt_coverage(
                id VARCHAR PRIMARY KEY,         -- hash of ticker + fetched_at date
                ticker VARCHAR NOT NULL,
                article_count INTEGER,
                top_sources VARCHAR,            -- JSON list of source domains
                top_countries VARCHAR,          -- JSON list of countries
                coverage_label VARCHAR,
                top_article_urls VARCHAR,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Create index for faster queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticker_date
            ON prices(ticker, date DESC)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_ticker_fetched
            ON news_articles(ticker, fetched_at DESC)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_gdelt_ticker_fetched
            ON gdelt_coverage(ticker, fetched_at DESC)
        """)
        logger.debug("Schema created/veried")

    def store(self, rows: List[OHLCVRow], replace: bool = True) -> int:
        """
        Store OHLCV data in database

        Args:
            rows: List of OHLCV rows to store
            replace: If True, replace existing data; if False, skip duplicates
        
        Returns:
            Number of rows inserted
        """
        if not rows:
            logger.warning("No rows to store")
            return 0
        
        ticker = rows[0].ticker

        if replace:
            # Delete existing data for this ticker's date range
            dates = [r.date for r in rows]
            min_date = min(dates)
            max_date = max(dates)

            self.conn.execute("""
                DELETE FROM prices
                WHERE ticker = ?
                AND date BETWEEN ? AND ?
                """, [ticker, min_date, max_date])
            
            logger.debug(
                f"Deleted existing {ticker} data from {min_date} to {max_date}"
            )
        
        # Insert data
        data = [
            (
                r.ticker, r.date, r.open, r.high, r.low, r.close, r.volume
            )
            for r in rows
        ]

        self.conn.executemany("""
            INSERT INTO prices (ticker, date, open, high, low, close, volume)
            VALUES(?,?,?,?,?,?,?)
            ON CONFLICT DO NOTHING
        """, data)
        
        logger.info(f"Stored {len(rows)} rows for {ticker}")
        return len(rows)
    
    def fetch(
            self,
            ticker: str,
            start_date: Optional[date] = None,
            end_date: Optional[date] = None,
            limit: Optional[int] = None
    ) -> List[OHLCVRow]:
        """
        Fetch OHLCV data from database.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (inclusive)
            end_date:
            limit: Maximum number of rows to return
        Returns:
            List of OHLCV rows, sorted by date descending
        """
        query = "SELECT ticker, date, open, high, low, close, volume FROM prices WHERE ticker = ?"
        params = [ticker.upper()]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
            
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date DESC"

        if limit:
            query += f" LIMIT {limit}"

        result = self.conn.execute(query, params).fetchall()

        rows = [
            OHLCVRow(
                ticker=row[0],
                date=row[1],
                open=row[2],
                high=row[3],
                low=row[4],
                close=row[5],
                volume=row[6]
            )
            for row in result
        ]
        
        logger.debug(f"Fetched {len(rows)} rows for {ticker}")
        return rows
    
    def get_latest(self, ticker: str) -> Optional[OHLCVRow]:
        """Get most recent price data for a ticker"""
        rows = self.fetch(ticker, limit=1)
        return rows[0] if rows else None
    
    def close(self) -> None:
        """Close database connection"""
        self.conn.close()
        logger.info("DuckDB connection closed")

    def has_fresh_news(self, ticker: str, max_age_hours: int = 24) -> bool:
        """
        Check if recent news articles exist for this ticker.
        Prevent unnecessary news fetches.

        Args:
            ticker: Stock ticker symbol
            max_age_hours: Maximum age of news in hours to consider "fresh"
        Returns:
            True if fresh news exists, False otherwise
        """
        result = self.conn.execute("""
            SELECT COUNT(*) FROM news_articles
            WHERE ticker = ?
            AND fetched_at >= NOW() - INTERVAL (? || ' hours')::INTERVAL
        """, [ticker.upper(), str(max_age_hours)]).fetchone()

        count = result[0] if result else 0
        logger.debug(f"Fresh news check for {ticker}: {count} articles found")
        return count > 0 
    
    def store_news(self, ticker: str, articles: list) -> int:
        """
        Store news articles in the database.

        Args:
            ticker: Stock ticker symbol
            articles: List of news article dictionaries with keys:
                id, title, url, source, published_at, sentiment_score, sentiment_label, topics

        Returns:
            Number of articles stored
        """
        if not articles:
            logger.warning(f"No articles to store for {ticker}")
            return 0

        data = [
            (
                a["id"],
                ticker.upper(),
                a.get("title"),
                a.get("url"),
                a.get("source"),
                a.get("published_at"),
                a.get("sentiment_score"),
                a.get("sentiment_label"),
                a.get("topics"), # stored as JSON string
                a.get("summary")
            )
            for a in articles
        ]

        self.conn.executemany("""
            INSERT INTO news_articles (
                id, ticker, title, url, source, published_at,
                sentiment_score, sentiment_label, topics, summary)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT DO NOTHING
        """, data)

        logger.info(f"Stored {len(articles)} news articles for {ticker}")
        return len(articles)
    
    def fetch_news(
            self, 
            ticker: str, 
            max_age_hours: int = 24, 
            limit: int = 20
        )-> list:
        """
        Fetch cached news articles for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            max_age_hours: Maximum age of news in hours to fetch
            limit: Maximum number of articles to return
        Returns:
            List of news article dictionaries, sorted by published_at descending
        """
        result = self.conn.execute("""
            SELECT id, title, url, source, published_at, sentiment_score, sentiment_label, topics, summary
            FROM news_articles
            WHERE ticker = ?
            AND fetched_at >= NOW() - INTERVAL (? || ' hours')::INTERVAL
            ORDER BY published_at DESC
            LIMIT ?
            """, [ticker.upper(), str(max_age_hours), limit]).fetchall()
        
        articles = [
            {
                "id": row[0],
                "title": row[1],
                "url": row[2],
                "source": row[3],
                "published_at": str(row[4]),
                "sentiment_score": row[5],
                "sentiment_label": row[6],
                "topics": row[7], # JSON string
                "summary": row[8],
            }
            for row in result
        ]
        logger.debug(f"Fetched {len(articles)} cached news for {ticker}")
        return articles
    
    def has_fresh_coverage(self, ticker: str, max_age_hours: int = 12) -> bool:
        """
        Check if recent GDELT coverage data exists for this ticker.
        Prevent unnecessary GDELT fetches.

        Args:
            ticker: Stock ticker symbol
            max_age_hours: Maximum age of coverage in hours to consider "fresh"
        Returns:
            True if fresh coverage exists, False otherwise
        """
        result = self.conn.execute("""
            SELECT COUNT(*) FROM gdelt_coverage
            WHERE ticker = ?
            AND fetched_at >= NOW() - INTERVAL (? || ' hours')::INTERVAL
        """, [ticker.upper(), str(max_age_hours)]).fetchone()

        count = result[0] if result else 0
        logger.debug(f"Fresh GDELT coverage check for {ticker}: {count} records found")
        return count > 0
    
    def store_coverage(self, ticker: str, coverage: dict) -> None:
        """
        Store GDELT coverage summary data in the database.

        Args:
            ticker: Stock ticker symbol
            coverage: Dictionary with keys:
                article_count, top_sources, top_countries, coverage_label
        """
        coverage_id = hashlib.md5(
            f"{ticker}{datetime.now().date()}".encode()
        ).hexdigest()[:16]

        self.conn.execute("""
            INSERT INTO gdelt_coverage (
                id, ticker, article_count, 
                top_sources, top_countries, coverage_label, top_article_urls)
            VALUES(?,?,?,?,?,?,?)
            ON CONFLICT DO NOTHING
        """, [
            coverage_id,
            ticker.upper(),
            coverage.get("article_count"),
            json.dumps(coverage.get("top_sources", [])),
            json.dumps(coverage.get("top_countries", [])),
            coverage.get("coverage_label", "Unknown"),
            json.dumps(coverage.get("top_article_urls", []))
        ])
        logger.info(f"Stored GDELT coverage for {ticker}")

    def fetch_coverage(self, ticker: str, max_age_hours: int = 12) -> Optional[dict]:
        """
        Fetch latest GDELT coverage summary for a ticker.

        Args:
            ticker: Stock ticker symbol
            max_age_hours: Maximum age of coverage in hours to fetch
        Returns:
            Coverage dictionary or None if not found
        """
        result = self.conn.execute("""
            SELECT article_count, top_sources, top_countries, coverage_label, top_article_urls, fetched_at
            FROM gdelt_coverage
            WHERE ticker = ?
            AND fetched_at >= NOW() - INTERVAL (? || ' hours')::INTERVAL
            ORDER BY fetched_at DESC
            LIMIT 1
        """, [ticker.upper(), str(max_age_hours)]).fetchone()
        
        if not result:
            return None
        
        return {
            "article_count": result[0],
            "top_sources": json.loads(result[1]),
            "top_countries": json.loads(result[2]),
            "coverage_label": result[3],
            "top_article_urls":  json.loads(result[4]),
            "fetched_at": str(result[5])
        }