"""
Evidence Normalizer
-------------------
Converts raw tool output into clean, typed, deduplicated EvidenceItems.
Normalized metadata is stored directly on EvidenceItem fields (not hidden in raw)

Design principles:
- One normalizer per source type
- Separate dedupe_hash (content fingerprint) from id (storage identity)
- Timezone-aware UTC timestamps 
- Source-specific quality validation before normalization
- Never crash — return None on bad input, caller handles it
- NORMALIZER_REGISTRY dispatch table for easy source expansion
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from src.models.schemas import EvidenceItem
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ==========================================================
# CONSTANTS / ENUMS
# ==========================================================

class SourceName(str, Enum):
    YAHOO_FINANCE = "yahoo_finance"
    TECHNICAL_ANALYSIS = "technical_analysis"
    ALPHA_VANTAGE_NEWS = "alpha_vantage_news"
    GDELT_COVERAGE = "gdelt_coverage"

class EvidenceType(str, Enum):
    PRICE_DATA = "PRICE_DATA"
    TECHNICAL_ANALYSIS = "TECHNICAL_ANALYSIS"
    NEWS_SENTIMENT = "NEWS_SENTIMENT"
    NEWS_COVERAGE = "NEWS_COVERAGE"

class DirectionalImpact(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"

MIN_CONFIDENCE = 0.15


# ==========================================================
# TIME HELPERS
# ==========================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def parse_datetime(value: Any) -> Optional[datetime]:
    """Best-effort parser for ISO timestamps or datetime objects,"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisocalendar(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None

# ==========================================================
# DEDUPLICATION
# ==========================================================

def _make_dedupe_hash(source: str, entity: str, content_signature: str) -> str:
    """Stable content fingerprint for duplicate detection."""
    raw = f"{source}::{entity}::{content_signature}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def _make_storage_id(source: str, entity: str, dedupe_hash: str, retrieved_at: datetime) -> str:
    """
    Unique storage ID for this evidence object.
    Allows multiple time-separated evidence items that share the same core content.
    """ 
    raw  = f"{source}::{entity}::{dedupe_hash}::{retrieved_at.isoformat()}"
    return "E_" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

def is_duplicate(dedupe_hash: str, existing_dedupe_hashes: Sequence[str]) -> bool:
    return dedupe_hash in set(existing_dedupe_hashes)

# ==========================================================
# UTILITY HELPERS
# ==========================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try: 
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    
def _compact_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())

def _passes_quality_gate(confidence: float, claim: str) -> bool:
    if confidence < MIN_CONFIDENCE:
        return False
    if not claim or len(_compact_text(claim)) < 20:
        return False
    return True

# ==========================================================
# DIRECTIONAL IMPACT HELPER
# ==========================================================

def _infer_directional_impact(signal: str) -> DirectionalImpact:
    """
    Map raw signal strings to a standard directional label.
    Returns: BULLISH | BEARISH | NEUTRAL | MIXED | UNKOWN
    """
    signal = (signal or "").lower()

    bullish_terms = {"bullish", "buy", "outperform", "positive", "strong", "uptrend", "above"}
    bearish_terms = {"bearish", "sell", "underperform", "negative", "weak", "downtrend", "below"}
    neutral_terms = {"neutral", "hold", "sideways", "stable"}

    if any(t in signal for t in {"mixed"}):
        return DirectionalImpact.MIXED
    if any(t in signal for t in bullish_terms):
        return DirectionalImpact.BULLISH
    if any(t in signal for t in bearish_terms):
        return DirectionalImpact.BEARISH
    if any(t in signal for t in neutral_terms):
        return DirectionalImpact.NEUTRAL
    return DirectionalImpact.UNKNOWN

def _serialize_for_raw(value: Any) -> Any:
    """Safe conversion for raw payload embedding."""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    return value


# ==========================================================
# CENTRALIZED EVIDENCE BUILDER
# ==========================================================
 
def _build_evidence_item(
    *,
    source: SourceName,
    ticker: str,
    claim: str,
    summary: str,
    key_claims: List[str],
    directional_impact: DirectionalImpact,
    evidence_type: EvidenceType,
    confidence: float,
    raw: Dict[str, Any],
    content_signature: str,
    retrieved_at: Optional[datetime] = None,
    published_at: Optional[datetime] = None,
    extra_raw_fields: Optional[Dict[str, Any]] = None,
    source_url: Optional[str] = None,
    ) -> EvidenceItem:
    """
    Centralized builder - keeps normalized structure consistent across all sources.
    Populates EvidenceItem fields directly (no longer hidden in raw["_normalized"]).
    """
    retrieved_at = retrieved_at or utc_now()
    dedupe_hash = _make_dedupe_hash(source.value, ticker, content_signature)
    evidence_id = _make_storage_id(source.value, ticker, dedupe_hash, retrieved_at)

    return EvidenceItem(
        # Identity
        id=evidence_id,
        claim=claim,

        # Source metadata
        source=source.value,
        entity=ticker,
        evidence_type = evidence_type.value,

        # Structured content - now first-class fields
        summary=summary,
        key_claims=key_claims,
        directional_impact=directional_impact.value,

        # Timestamps
        timestamp=retrieved_at,
        published_at=published_at,

        # Quality
        confidence=confidence,
        dedupe_hash=dedupe_hash,
        
        # Raw payload kept for full fidelity + extra source-specific fields
        raw={
            **raw,
            **(extra_raw_fields or {}),
        },
        source_url=source_url,
    )

# ==========================================================
# SOURCE-SPECIFIC QUALITY VALIDATORS
# ==========================================================

def _validate_price_raw(raw: Dict[str, Any]) -> bool:
    return(
        bool(raw.get("success"))
        and raw.get("end_price") is not None
    )

def _validate_technical_raw(raw: Dict[str, Any]) -> bool:
    return bool(raw.get("success")) and isinstance(raw.get("indicators"), dict)

def _validate_alpha_vantage_raw(raw: Dict[str, Any]) -> bool:
    return bool(raw.get("success")) and _safe_int(raw.get("article_count")) > 0

def _validate_gdelt_raw(raw: Dict[str, Any]) -> bool:
    coverage = raw.get("coverage", {}) or {}
    return bool(raw.get("success")) and _safe_int(coverage.get("article_count")) > 0

# ==========================================================
# NORMALIZERS — one per source type
# ==========================================================

def normalize_price_data(
    raw: Dict[str, Any],
    ticker: str, 
    existing_dedupe_hashes: Sequence[str],
) -> Optional[EvidenceItem]:
    """
    Normalize Yahoo Finance OHLCV output.
    Expected raw keys: 
        success, evidence_claim, ticker,
        period_days, start_price, end_price,
        price_change_pct, avg_volume, end_date (optional)
    """
    try:
        if not _validate_price_raw(raw):
            return None
        
        start = _safe_float(raw.get("start_price"))
        end = _safe_float(raw.get("end_price"))
        change = _safe_float(raw.get("price_change_pct"))
        volume = _safe_float(raw.get("avg_volume"))
        days = _safe_float(raw.get("period_days", 90), default=90)
        end_date_str = _compact_text(raw.get("end_date")) or "unknown_end_date"
        
        direction = (
            DirectionalImpact.BULLISH if change > 2
            else DirectionalImpact.BEARISH if change < -2
            else DirectionalImpact.NEUTRAL
        )

        summary = (
            f"{ticker} price moved from ${start:.2f} to ${end:.2f} "
            f"({change:+.1f}%) over {days} days. "
            f"Average daily volume: {volume:,.0f} shares."
        )

        key_claims = [
            f"Price change: {change:+1f}% over {days} days",
            f"End price: ${end:.2f}",
            f"Average volume: {volume:,.0f}",
        ]

        confidence = 0.90 # High - real exchange data
        claim = _compact_text(raw.get("evidence_claim")) or summary
        
        if not _passes_quality_gate(confidence, claim):
            return None
        
        content_signature = (
            f"price::{days}::{end_date_str}::{round(change, 2)}::{round(end, 2)}"
        )
        dedupe_hash = _make_dedupe_hash(SourceName.YAHOO_FINANCE.value, ticker, content_signature)
        if is_duplicate(dedupe_hash, existing_dedupe_hashes):
            logger.debug("[normalizer] Skipping duplicate price evidence for %s", ticker)
            return None
        
        published_at = parse_datetime(raw.get("end_date"))

        return _build_evidence_item(
            source=SourceName.YAHOO_FINANCE,
            ticker=ticker,
            claim=claim,
            summary=summary,
            key_claims=key_claims,
            directional_impact=direction,
            evidence_type=EvidenceType.PRICE_DATA,
            confidence=confidence,
            raw = raw,
            content_signature=content_signature,
            published_at=published_at,
            extra_raw_fields={
                "period_days": days,
                "start_price": start,
                "end_price": end,
                "price_change_pct": change,
                "avg_volume": volume,
            },
        )
    
    except Exception as e:
        logger.warning(f"[normalizer] price_data failed for {ticker}: {e}")
        return None
    
def normalize_technical_indicators(
    raw: Dict[str, Any],
    ticker: str,
    existing_dedupe_hashes: Sequence[str],
    ) -> Optional[EvidenceItem]:
    """
    Normalize technical analysis output.
    Expected raw keys: 
        success, indicators (dict),
        volume_analysis (optional), benchmark_comparison (optional),
        latest_bar_date(optional)
    """
    try:
        if not _validate_technical_raw(raw):
            return None
        
        indicators = raw.get("indicators", {}) or {}
        volume_analysis = raw.get("volume_analyis", {}) or {}
        benchmark = raw.get("benchmark_comparison", {}) or {}

        # Collect key claims
        key_claims: List[str] = []
        signal_texts: List[str] = []
        
        # RSI
        rsi_block = indicators.get("rsi_14", {}) or {}
        if rsi_block.get("current") is not None:
            rsi = _safe_float(rsi_block.get("current"))
            rsi_signal = _compact_text(rsi_block.get("signal"))
            key_claims.append(f"RSI(14): {rsi:.1f} ({rsi_signal or 'no signal'})")
            if rsi_signal:
                signal_texts.append(rsi_signal)
        
        # MACD
        macd_block = indicators.get("macd", {}) or {}
        if macd_block.get("histogram") is not None:
            hist   = _safe_float(macd_block.get("histogram"))
            interp = _compact_text(macd_block.get("interpretation"))
            key_claims.append(f"MACD histogram: {hist:.3f} ({interp or 'no interpretation'})")
            if interp:
                signal_texts.append(interp)
 
        # Trend
        trend_block = indicators.get("trend", {}) or {}
        if trend_block:
            pct          = _safe_float(trend_block.get("percent_from_sma"))
            trend_signal = _compact_text(trend_block.get("signal"))
            key_claims.append(f"Price vs SMA(20): {pct:+.1f}% ({trend_signal or 'no signal'})")
            if trend_signal:
                signal_texts.append(trend_signal)
 
        # Bollinger Bands
        bbands_block = indicators.get("bbands", {}) or {}
        if bbands_block:
            position = _compact_text(bbands_block.get("position"))
            squeeze  = bool(bbands_block.get("squeeze"))
            squeeze_text = " with squeeze" if squeeze else ""
            if position:
                key_claims.append(f"Bollinger position: {position}{squeeze_text}")
 
        # Volume
        if volume_analysis:
            ratio      = _safe_float(volume_analysis.get("volume_ratio"), default=1.0)
            vol_signal = _compact_text(volume_analysis.get("signal"))
            key_claims.append(f"Volume: {ratio:.1f}x average ({vol_signal or 'no signal'})")
            if vol_signal:
                signal_texts.append(vol_signal)
 
        # Benchmark
        if benchmark and "error" not in benchmark:
            rel            = _safe_float(benchmark.get("relative_strength"))
            benchmark_name = _compact_text(benchmark.get("benchmark")) or "SPY"
            key_claims.append(f"Relative strength vs {benchmark_name}: {rel:+.1f}%")
 
        if len(key_claims) < 2:
            return None

        # Aggregate direction from signals
        combined_signal = " ".join(signal_texts)
        direction = _infer_directional_impact(combined_signal)
        summary = f"{ticker} technical picture: " + "; ".join(key_claims[:4])
        claim = summary
        confidence = 0.85

        if not _passes_quality_gate(confidence, claim):
            return None
        
        latest_bar_date = _compact_text(raw.get("latest_bar_date")) or "unknown_bar"
        content_signature = f"technicals::{latest_bar_date}::{'|'.join(key_claims[:5])}"
        dedupe_hash = _make_dedupe_hash(SourceName.TECHNICAL_ANALYSIS.value, ticker, content_signature)

        if is_duplicate(dedupe_hash, existing_dedupe_hashes):
            logger.debug("[normalizer] Skipping duplicate technical evidence for %s")
            return None

        published_at = parse_datetime(raw.get("latest_bar_date"))

        return _build_evidence_item(
            source=SourceName.TECHNICAL_ANALYSIS,
            ticker=ticker,
            claim=claim,
            summary=summary,
            key_claims=key_claims,
            directional_impact=direction,
            evidence_type=EvidenceType.TECHNICAL_ANALYSIS,
            confidence=confidence,
            raw=raw,
            content_signature=content_signature,
            published_at=published_at,
            extra_raw_fields={
                "indicator_count": len(key_claims),
                "signal_count": len(signal_texts),
            },
        )
    
    except Exception as e:
        logger.warning(f"[normalizer] technical_indicators failed for {ticker}: {e}")
        return None
    
def normalize_alpha_vantage_news(
    raw: Dict[str, Any],
    ticker: str,
    existing_dedupe_hashes: Sequence[str],
    ) -> Optional[EvidenceItem]:
    """
    Normalize Alpha Vantage news sentiment output.

    Expected raw keys: 
        success, evidence_claim, article_count,
        aggregate_sentiment, sentiment_score, articles
    """
    try:
        if not _validate_alpha_vantage_raw(raw):
            return None
        
        article_count = _safe_int(raw.get("article_count"))
        aggregate_sentiment = _compact_text(raw.get("aggregate_sentiment")) or "Neutral"
        sentiment_score = _safe_float(raw.get("sentiment_score"))
        articles = raw.get("articles", []) or []

        if article_count == 0:
            return None
        
        key_claims: List[str] = []
        latest_published_at: Optional[datetime] = None

        for article in articles[:3]:
            title = _compact_text(article.get("title"))
            score = _safe_float(article.get("sentiment_score"))
            published_at=parse_datetime(
                article.get("time_published") or article.get("published_at")
            )
            if published_at and (
                latest_published_at is None or published_at > latest_published_at
            ):
                latest_published_at = published_at
            if title:
                key_claims.append(f"{title[:120]} (sentiment: {score:+.2f})")
     
        direction  = _infer_directional_impact(aggregate_sentiment)
        # Confidence scales with articles count
        confidence = round(min(0.45 + (article_count / 120.0), 0.85), 2)

        summary = (
            f"{ticker} news sentiment: {aggregate_sentiment} "
            f"(score: {sentiment_score:+.2f}) across {article_count} relevant articles."
        )

        claim = _compact_text(raw.get("evidence_claim")) or summary

        if not _passes_quality_gate(confidence, claim):
            return None
        latest_marker     = latest_published_at.isoformat() if latest_published_at else "no_pubdate"
        content_signature = (
            f"news_sentiment::{article_count}::{aggregate_sentiment.lower()}::"
            f"{round(sentiment_score, 3)}::{latest_marker}"
        )
        dedupe_hash = _make_dedupe_hash(SourceName.ALPHA_VANTAGE_NEWS.value, ticker, content_signature)
 
        if is_duplicate(dedupe_hash, existing_dedupe_hashes):
            logger.debug("[normalizer] Skipping duplicate Alpha Vantage news for %s", ticker)
            return None
        
        return _build_evidence_item(
            source=SourceName.ALPHA_VANTAGE_NEWS,
            ticker=ticker,
            claim=claim,
            summary=summary,
            key_claims=key_claims,
            directional_impact=direction,
            evidence_type=EvidenceType.NEWS_SENTIMENT,
            confidence=confidence,
            raw=raw,
            content_signature=content_signature,
            published_at=latest_published_at,
            extra_raw_fields={
                "article_count": article_count,
                "sentiment_score": sentiment_score,
                "aggregate_sentiment": aggregate_sentiment,
            },
        )
    except Exception as e:
        logger.warning(f"[normalizer] alpha_vantage_news failed for {ticker}: {e}")
        return None
    

def normalize_gdelt_news(
    raw: Dict[str, Any],
    ticker: str,
    existing_dedupe_hashes: Sequence[str],
    ) -> Optional[EvidenceItem]:
    """
    Normalize GDELT global coverage output.
    Expected raw keys: success, evidence_claim, coverage (dict with article_count,
                        top_domains, top_countries)
    """
    try:
        if not _validate_gdelt_raw(raw):
            return None
 
        coverage      = raw.get("coverage", {}) or {}
        article_count = _safe_int(coverage.get("article_count"))
        top_domains   = coverage.get("top_domains", []) or []
        top_countries = coverage.get("top_countries", []) or []
 
        if article_count == 0:
            return None
 
        key_claims = [f"Global coverage volume: {article_count} articles"]
        if top_domains:
            key_claims.append(f"Top domains: {', '.join(map(str, top_domains[:3]))}")
        if top_countries:
            key_claims.append(f"Top countries: {', '.join(map(str, top_countries[:3]))}")
 
        direction  = DirectionalImpact.NEUTRAL
        confidence = round(min(0.40 + (article_count / 250.0), 0.75), 2)
 
        summary = (
            f"{ticker} global media coverage: {article_count} articles. "
            f"Top domains: {', '.join(map(str, top_domains[:2])) if top_domains else 'N/A'}."
        )
        claim = _compact_text(raw.get("evidence_claim")) or summary
 
        if not _passes_quality_gate(confidence, claim):
            return None
 
        content_signature = (
            f"gdelt::{article_count}::"
            f"{'|'.join(map(str, top_domains[:3]))}::"
            f"{'|'.join(map(str, top_countries[:3]))}"
        )
        dedupe_hash = _make_dedupe_hash(SourceName.GDELT_COVERAGE.value, ticker, content_signature)
 
        if is_duplicate(dedupe_hash, existing_dedupe_hashes):
            logger.debug("[normalizer] Skipping duplicate GDELT coverage for %s", ticker)
            return None
 
        return _build_evidence_item(
            source=SourceName.GDELT_COVERAGE,
            ticker=ticker,
            claim=claim,
            summary=summary,
            key_claims=key_claims,
            directional_impact=direction,
            evidence_type=EvidenceType.NEWS_COVERAGE,
            confidence=confidence,
            raw=raw,
            content_signature=content_signature,
            extra_raw_fields={
                "article_count": article_count,
                "top_domains": top_domains[:5],
                "top_countries": top_countries[:5],
            },
        )
 
    except Exception as e:
        logger.warning("[normalizer] gdelt_news failed for %s: %s", ticker, e)
        return None
    
# ==========================================================
# DISPATCH TABLE — used by Researcher agent
# ==========================================================

NORMALIZER_REGISTRY: Dict[str, Any] = {
    SourceName.YAHOO_FINANCE.value:       normalize_price_data,
    SourceName.TECHNICAL_ANALYSIS.value:  normalize_technical_indicators,
    SourceName.ALPHA_VANTAGE_NEWS.value:  normalize_alpha_vantage_news,
    SourceName.GDELT_COVERAGE.value:      normalize_gdelt_news,
}

def normalize(
    source: str,
    raw: Dict[str, Any],
    ticker: str,
    existing_dedupe_hashes: Sequence[str]
) -> Optional[EvidenceItem]:
    """
    Main dispatch function used by the Researcher agent.
    Args:
        source:                 Source identifier string (must match SourceName enum value)
        raw:                    Raw dict returned by the tool function
        ticker:                 Primary ticker under analysis
        existing_dedupe_hashes: List of dedupe hashes already in the evidence vault
                                (use ResearchCycleState.vault_dedupe_hashes())
    Returns:
    Normalized EvidenceItem, or None if:
    - source not registered
    - data fails quality gate
    - normalizer raises an error
    """
    normalize_fn = NORMALIZER_REGISTRY.get(source)

    if not normalize_fn:
        logger.warning("[normalizer] No normalizer registered for source: '%s'", source)
        return None
    
    return normalize_fn(
        raw=raw,
        ticker=ticker,
        existing_dedupe_hashes=existing_dedupe_hashes,
    )