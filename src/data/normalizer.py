"""
Evidence Normalizer
-------------------
Converts raw tool output into clean, typed, deduplicated EvidenceItem
Each normalizer function receives a raw result dict and returns an EvidenceItem
(or None if the data is too low quality to include)

Design principles:
- One normalizer per source type
- Always extract: summary, key_claims, directional_impact, confidence
- Dedup check against existing vault IDs
- Never crash - return None on bad input, caller handles it
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.models.schemas import EvidenceItem
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ==========================================================
# DEDUPLICATION
# ==========================================================

def _make_evidence_id(source: str, entity: str, content_key: str) -> str:
    """
    Generate a stable, deterministic evidence ID.
    Same source + entity + content = same ID -> natural dedup.
    """ 
    raw  = f"{source}::{entity}::{content_key}"
    return "E_" + hashlib.md5(raw.encode()).hexdigest()[:8]

def is_duplicate(evidence_id: str, vault_ids: List[str]) -> bool:
    return evidence_id in vault_ids


# ==========================================================
# QUALITY GATE
# ==========================================================

MIN_CONFIDENCE = 0.15 # Below this -> reject entirely

def _passes_quality_gate(confidence: float, claim: str) -> bool:
    if confidence < MIN_CONFIDENCE:
        return False
    if not claim or len(claim.strip()) < 20:
        return False
    return True

# ==========================================================
# DIRECTIONAL IMPACT HELPER
# ==========================================================

def _infer_directional_impact(signal: str) -> str:
    """
    Map raw signal strings to a standard directional label.
    Returns: BULLISH | BEARISH | NEUTRAL | MIXED | UNKOWN
    """
    signal = (signal or "").lower()

    bullish_terms = {"bullish", "buy", "outperform", "positive", "strong", "uptrend", "above"}
    bearish_terms = {"bearish", "sell", "underperform", "negative", "weak", "downtrend", "below"}
    neutral_terms = {"neutral", "hold", "mixed", "sideways", "stable"}

    if any(t in signal for t in bullish_terms):
        return "BULLISH"
    if any(t in signal for t in bearish_terms):
        return "BEARISH"
    if any(t in signal for t in neutral_terms):
        return "NEUTRAL"
    return "UNKNOWN"


# ==========================================================
# NORMALIZERS — one per source type
# ==========================================================

def normalize_price_data(
    raw: Dict[str, Any],
    ticker: str, 
    vault_ids: List[str]
) -> Optional[EvidenceItem]:
    """
    Normalize Yahoo Finance OHLCV output.
    Expected raw keys: success, evidence_claim, ticker, period_days,
                        start_price, end_price, price_change_pct, avg_volume
    """
    try:
        if not raw.get("success"):
            return None
        
        evidence_id = _make_evidence_id("yahoo_finance", ticker, "price_90d")
        if is_duplicate(evidence_id, vault_ids):
            logger.debug(f"[normalizer] Skipping duplicate: {evidence_id}")
            return None
        
        start = raw.get("start_price", 0)
        end = raw.get("end_price", 0)
        change = raw.get("price_change_pct", 0)
        volume = raw.get("avg_volume", 0)
        days = raw.get("period_days", 90)
        
        direction = "BULLISH" if change > 2 else "BEARISH" if change < -2 else "NEUTRAL"

        summary = (
            f"{ticker} price moved from ${start:.2f} to ${end:.2f} "
            f"({change:+.1f}%) over {days} days. "
            f"Average daily volume: {volume:,.0f} shares."
        )

        key_claims = [
            f"Price change: {change:+1f}% over {days} days",
            f"Current price: ${end:.2f}",
            f"Average volume: {volume:,.0f}",
        ]

        confidence = 0.9 # High - real exchange data
        claim = raw.get("evidence_claim", summary)
        
        if not _passes_quality_gate(confidence, claim):
            return None
        
        return EvidenceItem(
            id=evidence_id,
            claim=claim,
            source="yahoo_finance",
            timestamp=datetime.now(),
            confidence=confidence,
            raw = {
                **raw,
                "_normalized": {
                    "summary": summary,
                    "key_claims": key_claims,
                    "directional_impact": direction,
                    "entity": ticker,
                    "evidence_type": "PRICE_DATA"
                }
            }
        )
    
    except Exception as e:
        logger.warning(f"[normalizer] price_data failed for {ticker}: {e}")
        return None
    
def normalize_technical_indicators(
    raw: Dict[str, Any],
    ticker: str,
    vault_ids: List[str]
) -> Optional[EvidenceItem]:
    """
    Normalize technical analysis output.
    Expected raw keys: success, indicators (dict), volume_analysis,
                        benchmark_comparison
    """
    try:
        if not raw.get("success"):
            return None
        
        evidence_id = _make_evidence_id("technical_analysis", ticker, "indicators")
        if is_duplicate(evidence_id, vault_ids):
            logger.debug(f"[normalizer] Skipping duplicate: {evidence_id}")
            return None
        
        ind = raw.get("indicators", {})
        vol = raw.get("volume_analysis", {})
        bench = raw.get("benchmark_comparison", {})

        # Collect key claims
        key_claims = []
        signals = []
        
        if "rsi_14" in ind and ind["rsi_14"].get("current"):
            rsi = ind["rsi_14"]["current"]
            sig = ind["rsi_14"].get("signal", "")
            key_claims.append(f"RSI(14): {rsi:.1f} - {sig}")
            signals.append(sig)
        
        if "macd" in ind and ind["macd"].get("histogram") is not None:
            hist = ind["macd"]["histogram"]
            interp = ind["macd"].get("interpretation", "")
            key_claims.append(f"MACD histogram: {hist:.3f} ({interp})")
            signals.append(interp)

        if "trend" in ind:
            trend = ind["trend"]
            pct = trend.get("percent_from_sma", 0)
            sig = trend.get("signal", "")
            key_claims.append(f"Price {pct:+.1f}% from SMA(20) - {sig}")
            signals.append(sig)
        
        if "bbands" in ind:
            bb = ind["bbands"]
            pos = bb.get("position", "")
            squeeze = " (squeeze)" if bb.get("squeeze") else ""
            key_claims.append(f"Bollinger: {pos}{squeeze}")

        if vol:
            ratio = vol.get("volume_ratio", 1.0)
            sig = vol.get("signal", "")
            key_claims.append(f"Volume: {ratio:.1f}x average ({sig})")

        
        if bench and "error" not in bench:
            rel = bench.get("relative_strength", 0)
            bmark = bench.get("benchmark", "SPY")
            key_claims.append(f"vs {bmark}: {rel:+.1f}% relative strength")

        # Aggregate direction from signals
        combined_signal = " ".join(signals).lower()
        direction = _infer_directional_impact(combined_signal)

        summary = f"{ticker} technical signals: " + "; ".join(key_claims[:4])
        confidence = 0.85
        
        # Build a clean claim string (reuse existing if present)
        claim = summary

        if not _passes_quality_gate(confidence, claim):
            return None
        
        return EvidenceItem(
            id=evidence_id,
            claim=claim,
            source="technical_analysis",
            timestamp=datetime.now(),
            confidence=confidence,
            raw={
                **raw,
                "_normalized": {
                    "summary": summary,
                    "key_claims": key_claims,
                    "directional_impact": direction,
                    "entity": ticker,
                    "evidence_type": "TECHNICAL_ANALYSIS",
                }
            }
        )
    
    except Exception as e:
        logger.warning(f"[normalizer] technical_indicators failed for {ticker}: {e}")
        return None
    
def normalize_alpha_vantage_news(
    raw: Dict[str, Any],
    ticker: str,
    vault_ids: List[str]
) -> Optional[EvidenceItem]:
    """
    Normalize Alpha Vantage news sentiment output.
    Expected raw keys: success, evidence_claim, article_count,
                        aggregate_sentiment, sentiment_score, articles
    """
    try:
        if not raw.get("success"):
            return None
        
        evidence_id = _make_evidence_id("alpha_vantage_news", ticker, "sentiment")
        if is_duplicate(evidence_id, vault_ids):
            logger.debug(f"[normalizer] Skipping duplicate: {evidence_id}")
            return None
        
        article_count = raw.get("article_count", 0)
        agg_sentiment = raw.get("aggregate_sentiment", "Neutral")
        sentiment_score = raw.get("sentiment_score", 0.0)

        if article_count == 0:
            return None
        
        direction = _infer_directional_impact(agg_sentiment)

        # Top headlines as key claims
        articles = raw.get("articles", [])
        key_claims = []
        for a in articles[:3]:
            title = a.get("title", "")
            score = a.get("sentiment_score", 0)
            if title:
                key_claims.append(f"{title[:80]} (sentiment: {score:+.2f})")
     
        # Confidence scales with articles count
        confidence = round(min(0.5 + (article_count / 100), 0.9), 2)

        summary = (
            f"{ticker} news sentiment: {agg_sentiment} "
            f"(score: {sentiment_score:+.2f}) across {article_count} articles."
        )

        claim = raw.get("evidence_claim", summary)

        if not _passes_quality_gate(confidence, claim):
            return None
        
        return EvidenceItem(
            id=evidence_id,
            claim=claim,
            source="alpha_vantage_news",
            timestamp=datetime.now(),
            confidence=confidence,
            raw={
                **raw,
                "_normalized": {
                    "summary": summary,
                    "key_claims": key_claims,
                    "directional_impact": direction,
                    "entity": ticker,
                    "evidence_type": "NEWS_SEARCH",
                    "article_count": article_count,
                    "sentiment_score": sentiment_score,
                }
            }
        )
    except Exception as e:
        logger.warning(f"[normalizer] alpha_vantage_news failed for {ticker}: {e}")
        return None
    

def normalize_gdelt_news(
    raw: Dict[str, Any],
    ticker: str,
    vault_ids: List[str]
) -> Optional[EvidenceItem]:
    """
    Normalize GDELT global coverage output.
    Expected raw keys: success, evidence_claim, coverage (dict with article_count,
                        top_domains, top_countries)
    """
    try:
        if not raw.get("success"):
            return None
        
        evidence_id = _make_evidence_id("gdelt_coverage", ticker, "coverage")
        if is_duplicate(evidence_id, vault_ids):
            logger.debug(f"[normalizer] Skipping duplicate: {evidence_id}")
            return None
        
        coverage = raw.get("coverage", {})
        article_count = coverage.get("article_count", 0)
        top_domains = coverage.get("top_domains", [])
        top_countries = coverage.get("top_countries", [])

        if article_count == 0:
            return None
        
        key_claims = [
            f"Global article count: {article_count}"
        ]
        if top_domains:
            key_claims.append(f"Top sources: {', '.join(top_domains[:3])}")
        if top_countries:
            key_claims.append(f"Top countries: {', '.join(top_countries[:3])}")

        # GDELT is coverage volume - directional signal is weak
        direction = "NEUTRAL"

        confidence = round(min(0.4 + (article_count / 200), 0.75), 2)
        summary = (
            f"{ticker} global media coverage: {article_count} articles. "
            f"Top sources: {', '.join(top_domains[:2]) if top_domains else 'N/A'}."
        )

        claim = raw.get("evidence_claim", summary)

        if not _passes_quality_gate(confidence, claim):
            return None
        
        return EvidenceItem(id=evidence_id,
            claim=claim,
            source="gdelt_coverage",
            timestamp=datetime.now(),
            confidence=confidence,
            raw={
                **raw,
                "_normalized": {
                    "summary": summary,
                    "key_claims": key_claims,
                    "directional_impact": direction,
                    "entity": ticker,
                    "evidence_type": "NEWS_SEARCH",
                    "article_count": article_count,
                }
            }
        )
    
    except Exception as e:
        logger.warning(f"[normalizer] gdelt_news failed for {ticker}: {e}")
        return None
    

# ==========================================================
# DISPATCH TABLE — used by Researcher agent
# ==========================================================

NORMALIZER_REGISTRY = {
    "yahoo_finance":       normalize_price_data,
    "technical_analysis":  normalize_technical_indicators,
    "alpha_vantage_news":  normalize_alpha_vantage_news,
    "gdelt_coverage":      normalize_gdelt_news,
}

def normalize(
    source: str,
    raw: Dict[str, Any],
    ticker: str,
    vault_ids: List[str]
) -> Optional[EvidenceItem]:
    """
    Main dispatch function.
    Researcher calls this - it routes to the right normalizer.

    Returns None if:
    - source not registered
    - data fails quality gate
    - normalizer raises an error
    """
    normalize_fn = NORMALIZER_REGISTRY.get(source)

    if not normalize_fn:
        logger.warning(f"[normalizer] No normalizer registered for source: '{source}'")
        return None
    
    return normalize_fn(raw=raw, ticker=ticker, vault_ids=vault_ids)