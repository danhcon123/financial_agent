"""
LangChain tools for news and sentiment data fetching.
Sources: Alpha Vantage (news + sentiment), GDELT (global news tone)
"""
import hashlib
import json
import requests
from typing import Optional
from datetime import datetime

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.config.settings import Settings, get_settings
from src.data.storage import DuckDBStorage
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================================
# Input Schemas
# ============================================================================
class AlphaVantageNewsInput(BaseModel):
    """Input schema for Alpha Vantage news fetching"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    limit: int = Field(10, ge= 1, le= 50, description="Number of articles to fetch")


class GdeltNewsInput(BaseModel):
    """"Input schema for GDELT news fetching tool"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    company_name: Optional[str] = Field(
        None,
        description="Company name to search (e.g., 'Apple'). Auto-resolved from ticker if not provided."
    )
    limit: int = Field(10, ge=1, le=100, description="Number of articles to fetch")

# ============================================================================
# Tool Functions
# ============================================================================
def fetch_alpha_vantage_news(ticker: str, limit: int = 10) -> dict:
    """
    Fetch news articles and sentiment data for a given ticker from Alpha Vantage.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL)
        limit: Number of articles to fetch (max 50)
    Returns:
        Dict with success, articles, aggregate_sentiment, evidence_claim
    """
    ticker = ticker.upper()
    settings = get_settings()

    # ------------------------------------------------------------------
    # STEP 1: Check cache first (save API calls)
    # ------------------------------------------------------------------
    storage = DuckDBStorage()

    if storage.has_fresh_news(ticker, max_age_hours=24):
        logger.info(f"Cache hit: returning stored news for {ticker}")
        articles = storage.fetch_news(ticker, max_age_hours=24, limit=limit)
        storage.close()
        return {
            "success": True,
            "ticker": ticker,
            "articles": articles,
            "article_count": len(articles),
            "aggregate_sentiment": _compute_aggregate_sentiment(articles),
            "evidence_claim": _build_evidence_claim(ticker, articles)
        }
    
    # ------------------------------------------------------------------
    # STEP 2: Check API key exists
    # ------------------------------------------------------------------
    if not settings.alpha_vantage_api_key:
        logger.error("ALPHA_VANTAGE_API_KEY not set in .env")
        storage.close()
        return {
            "success": False,
            "ticker": ticker,
            "error": "Alpha Vantage API key not configured"
        }
    
    # ------------------------------------------------------------------
    # STEP 3: Call Alpha Vantage API
    # ------------------------------------------------------------------
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "limit": min(limit, 500),
            "apikey": settings.alpha_vantage_api_key
        }

        logger.info(f"Calling Alpha Vantage API for NEWS_SENTIMENT: {ticker}")
        response = requests.get(url, params = params, timeout = 10)
        response.raise_for_status()
        data = response.json()

        # ------------------------------------------------------------------
        # STEP 4: Check for API errors (rate limit, invalid key, etc.)
        # ------------------------------------------------------------------
        if "Information" in data:
            # Alpha Vantage returns this key when rate limited
            logger.warning(f"Alpha Vantage rate limit hit: {data['Information']}")
            storage.close()
            return {
                "success": False,
                "ticker": ticker,
                "error": f"Rate limited: {data['Information']}"
            }

        if "feed" not in data:
            logger.error(f"Unexpected API response structure: {list(data.keys())}")
            storage.close()
            return {
                "success": False,
                "ticker": ticker,
                "error": "No feed in response — check API key or ticker"
            }
        
        # ------------------------------------------------------------------
        # STEP 5: Parse articles
        # ------------------------------------------------------------------
        articles = []
        for item in data["feed"]:
            # Find this ticker's specific sentiment score in the article
            # (each article can mention multiple tickers)
            ticker_sentiment_score = 0.0
            for ts in item.get("ticker_sentiment", []):
                if ts.get("ticker") == ticker:
                    ticker_sentiment_score = float(ts.get("sentiment_score", 0.0))
                    break
            
            # Generate stable unique ID from ticker + url
            # This means:
            # - Same ticker + same URL → **always same ID** → `ON CONFLICT DO NOTHING` skips the duplicate 
            # - Different ticker + same URL → **different ID** → stored separately (AAPL news vs MSFT news) 
            # - Different URL → **different ID** → new article stored 

            article_id = hashlib.md5(
                f"{ticker}_{item['url']}".encode()
                ).hexdigest()[:16]
            
            # Parse published time: Alpha Vantage format is "20240315T143000"
            published_str = item.get("time_published", "")
            try:
                published_at = datetime.strptime(published_str, "%Y%m%dT%H%M%S")
            except ValueError:
                published_at = datetime.now()  # fallback to now if parsing fails

            # Extract topic labels as JSON string
            topics = json.dumps([ t.get("topic", "") for t in item.get("topics", [])])

            articles.append({
                "id": article_id,
                "title": item.get("title"),
                "url": item.get("url"),
                "source": item.get("source"),
                "summary": item.get("summary"),
                "published_at": published_at,
                "sentiment_score": ticker_sentiment_score,
                "sentiment_label": _parse_sentiment_label(ticker_sentiment_score),
                "topics": topics,
            })

        # ------------------------------------------------------------------
        # STEP 6: Store in DB for future cache hits
        # ------------------------------------------------------------------
        storage.store_news(ticker, articles)
        storage.close()

        logger.info(f"Fetched and stored {len(articles)} articles for {ticker} from Alpha Vantage")
        return {
            "success": True,
            "ticker": ticker,
            "source": "api",
            "articles": articles,
            "article_count": len(articles),
            "aggregate_sentiment": _compute_aggregate_sentiment(articles),
            "evidence_claim": _build_evidence_claim(ticker, articles)
        }
    
    except requests.exceptions.Timeout as e:
        logger.error(f"Alpha Vantage request timed out for {ticker}")
        storage.close()
        return {
            "success": False,
            "ticker": ticker,
            "error": "Request timed out"
        }
    
    except Exception as e:
        logger.exception(f"Error fetching news for {ticker} from Alpha Vantage: {str(e)}")
        storage.close()
        return {
            "success": False,
            "ticker": ticker,
            "error": str(e)
        }
    

def fetch_gdelt_news(
    ticker: str,
    company_name: Optional[str] = None,
    limit: int = 10
) -> dict:
    """
    Fetch global news coverage volume for a company from GDELT.
    Used as a coverage intensity signal, not article content.
    No API key required. Uses cache-first strategy.

    GDELT complements Alpha Vantage by providing:
    - Broader global source coverage
    - Non-US/non-English media perspective
    - No rate limits

    Args:
        ticker: Stock ticker symbol (e.g., AAPL)
        company_name: Optional company name for better search results (e.g., "Apple Inc.")
        limit: Number of articles to fetch (max 100)
    Returns:
        Dict with success, articles, aggregate_sentiment, evidence_claim
    """
    ticker = ticker.upper()
    query_name = _resolve_company_name(ticker, company_name)

    # ------------------------------------------------------------------
    # STEP 1: Check cache first (save API calls)
    # ------------------------------------------------------------------
    storage = DuckDBStorage()

    if storage.has_fresh_coverage(ticker, max_age_hours=12):
        logger.info(f"Cache hit: returning stored news for {ticker}")
        coverage  = storage.fetch_coverage(ticker, max_age_hours=12)
        storage.close()
        return {
            "success": True,
            "ticker": ticker,
            "source": "cache",
            "coverage": coverage,
            "evidence_claim": _build_gdelt_evidence_claim(ticker, query_name, coverage)
        }
        
    # ------------------------------------------------------------------
    # STEP 2: Call GDELT API
    # ------------------------------------------------------------------
    try:
        settings = get_settings()
        base_url = f"{settings.gdelt_base_url}/doc/doc"

        params = {
            "query": f'"{query_name}" sourcelang:english',
            "mode": "artlist",
            "maxrecords": min(limit, 250),
            "format": "json",
            "timespan": "24h"
        }

        logger.info(f"Calling GDELT API for news on {query_name} (ticker: {ticker})")
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        raw_articles = data.get("articles", [])

        # ------------------------------------------------------------------
        # STEP 3: Check for API errors
        # ------------------------------------------------------------------
        # Extract top domains
        domains = []
        countries = []
        for item in raw_articles:
            domain = item.get("domain", "")
            country = item.get("sourcecountry", "")
            if domain:
                domains.append(domain)
            if country:
                countries.append(country)
        # Count top 5 unique sources and countries
        from collections import Counter
        top_sources = [d for d, _ in Counter(domains).most_common(5)]
        top_countries = [c for c, _ in Counter(countries).most_common(5)]

        article_count = len(raw_articles)
        coverage_label = _compute_coverage_label(article_count)
        
        top_article_urls = [
            item.get("url", "")
            for item in raw_articles[:5]
            if item.get("url")
        ]

        coverage = {
            "article_count": article_count,
            "top_sources": top_sources,
            "top_countries": top_countries,
            "coverage_label": coverage_label,
            "top_article_urls": top_article_urls
        }

        # ------------------------------------------------------------------
        # STEP 4: Store in DB for future cache hits
        # ------------------------------------------------------------------
        storage.store_coverage(ticker, coverage)
        storage.close()

        logger.info(f"GDELT coverage for {ticker}: {coverage_label} ({article_count} articles)")
        return {
            "success": True,
            "ticker": ticker,
            "source": "api",
            "coverage": coverage,
            "evidence_claim": _build_gdelt_evidence_claim(ticker, query_name, coverage)
        }
    
    except requests.exceptions.Timeout:
        logger.error(f"GDELT request timed out for {ticker}")
        storage.close()
        return {"success": False, "ticker": ticker, "error": "GDELT request timed out"}

    except Exception as e:
        logger.exception(f"Failed to fetch GDELT news for {ticker}")
        storage.close()
        return {"success": False, "ticker": ticker, "error": str(e)}

# ============================================================================
# Helpers
# ============================================================================
def _parse_sentiment_label(score: float) -> str:
    """
    Convert Alpha Vantage sentiment score to readable label.
    
    Alpha Vantage scoring:
        score <= -0.35  → Bearish
        score <= -0.15  → Somewhat Bearish
        score <   0.15  → Neutral
        score <   0.35  → Somewhat Bullish
        score >= 0.35   → Bullish
    """
    if score <= -0.35:
        return "Bearish"
    elif score <= -0.15:
        return "Somewhat Bearish"
    elif score < 0.15:
        return "Neutral"
    elif score < 0.35:
        return "Somewhat Bullish"
    else:
        return "Bullish"
    
def _compute_aggregate_sentiment(articles: list) -> dict:
    """Compute average sentiment score and overall label across articles"""
    if not articles:
        return {"score": 0.0, "label": "Neutral", "article_count": 0}
    
    scores = [
        a["sentiment_score"] for a in articles
        if a.get("sentiment_score") is not None
    ]

    if not scores:
        return {"score": 0.0, "label": "Neutral", "article_count": len(articles)}
    
    avg_score = sum(scores) / len(scores)
    return {
        "score": round(avg_score, 4),
        "label": _parse_sentiment_label(avg_score),
        "article_count": len(articles)
    }

def _build_evidence_claim(ticker: str, articles: list) -> str:
    """"
    Build a concise evidence claim string for the analyst LLM.
    Example output:
        "AAPL news sentiment (10 articles): Bullish (avg score: 0.52).
        Top stories: 'Apple beats Q2 earnings', 'iPhone demand strong',
        'Apple expands AI in iOS 19'."
    """
    if not articles:
        return f"{ticker} news sentiment: No recent articles found."
    
    sentiment = _compute_aggregate_sentiment(articles)

    # Top 3 article headlines
    top_headlines = [
        a["title"] for a in articles[:3] if a.get("title")
    ]
    headlines_str = ", ".join(f"'{h}'" for h in top_headlines)
    
    urls = [a["url"] for a in articles[:3] if a.get("url")]
    urls_str = "\n -".join(urls) if urls else "None available"
    return (
        f"{ticker} news sentiment ({sentiment['article_count']} articles): "
        f"{sentiment['label']} (avg score: {sentiment['score']}). "
        f"Top stories: {headlines_str}."
        f"Reference URLs:\n - {urls_str}"
    )

def _resolve_company_name(ticker: str, company_name: Optional[str] = None) -> str:
    """
    Resolve company name from ticker for GDELT search query.
    
    Priority:
    1. Use-provided company_name (override)
    2. yfinance lookup (dynamic, works for any ticker)
    3. Fallback to ticker itself if yfinance fails
    """
    if company_name:
        return company_name
    
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        # yfinance returns 'shortName' or 'longName' for company name
        name = info.get("shortName") or info.get("longName") or ticker
        # Clean up common suffixes that confuse search engines
        # "Apple Inc." → "Apple", "Tesla, Inc." → "Tesla"
        for suffix in [", Inc.", " Inc.", " Corp.", " Ltd.", " LLC", " Co."]:
            name = name.replace(suffix, "")
        logger.debug(f"Resolved {ticker} → '{name}' via yfinance")
        return name.strip()
    
    except Exception as e:
        logger.warning(f"Could not resolve company name for {ticker}: {e}, using ticker as fallback")
        return ticker

def _compute_coverage_label(article_count: int) -> dict:
    """Convert article count to human-readable coverage intensity label"""
    if article_count == 0:
        return "No coverage"
    elif article_count >= 20:
        return "High coverage"
    elif article_count >= 8:
        return "Moderate coverage"
    else:
        return "Low coverage"

def _build_gdelt_evidence_claim(
        ticker: str,
        company_name: str,
        coverage: Optional[dict]
    ) -> str:
    """
    Build an evidence claim for GDELT coverage summary.
    Example:
        "Apple (AAPL) global news coverage: Moderate coverage
         (8 articles, last 24h). Sources: reuters.com, bbc.co.uk,
         ft.com. Top stories: 'Apple expands in India', ..."
    """
    if not coverage or coverage.get("article_count", 0) == 0:
        return f"No recent global news coverage found for {company_name} ({ticker})."
    
    sources_str = ", ".join(coverage.get("top_sources", [])[:3])
    countries_str = ", ".join(coverage.get("top_countries", [])[:3])

    # Build verifiable URL list
    urls = coverage.get("top_article_urls", [])
    urls_str = "\n  - ".join(urls) if urls else "None available"

    return (
        f"{company_name} ({ticker}) global media coverage: "
        f"{coverage['coverage_label']} ({coverage['article_count']} articles, last 24h). "
        f"Top sources: {sources_str}. "
        f"Top regions: {countries_str}. "
        f"Reference URLs:\n  - {urls_str}"
    )


# ============================================================================
# Tool Definitions
# ============================================================================
alpha_vantage_news_tool = StructuredTool.from_function(
    func=fetch_alpha_vantage_news,
    name="fetch_alpha_vantage_news",
    description=(
        "Fetches recent news articles and sentiment scores for a stock ticker "
        "from Alpha Vantage. Uses local cache to avoid redundant API calls. "
        "Returns article titles, summaries, sentiment scores, and an "
        "aggregate evidence claim for the analyst agent."
    ),
    args_schema=AlphaVantageNewsInput
)

gdelt_news_tool = StructuredTool.from_function(
    func=fetch_gdelt_news,
    name="fetch_gdelt_news",
    description=(
        "Fetches global news coverage for a company from GDELT — "
        "a free, no-key-required global news database. "
        "Complements Alpha Vantage by providing broader international "
        "source coverage. Returns article titles, domains, and a "
        "coverage intensity summary for the analyst agent."
    ),
    args_schema=GdeltNewsInput
)

# Exported tool list
NEWS_TOOLS = [alpha_vantage_news_tool, gdelt_news_tool]