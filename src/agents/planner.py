from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from src.models.enums import TaskStatus, TaskType
from src.models.schemas import (
    CriticOutput,
    EvidenceItem,
    ResearchCycleState,
    ResearchRequest,
    ResearchTask
)
from src.utils.json_parser import _extract_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ==========================================================
# RESULT OBJECT
# ==========================================================

@dataclass
class PlannerResult:
    """Structured result from Planner execution"""
    tasks: List[ResearchTask]
    planner_status: str # 'ok' | 'no_gaps' | 'budget_exhausted' | 'llm_failed'
    used_llm: bool
    used_fallback: bool
    notes: List[str]


# ==========================================================
# PROMPTS
# ==========================================================

PLANNER_SYSTEM_PROMPT = """\
You are a Research Planner for a financial analysis system.

Your job is to convert critic-identified evidence gaps into a small set of precise retrieval tasks.

Return ONLY a JSON array. No prose. No markdown.

Each item must follow this schema:
[
    {
        "task_type": "<one of: PRICE_DATA | TECHNICALS |NEWS_SEARCH | FUNDAMENTALS | PEER_COMPARE | EARNINGS_CHECK | FILING_SUMMARY>",
        "entity": "<ticker or company name>",
        "question": "<specific question answerable by a single retrieval or calculation step>",
        "priority": <1 | 2 | 3>,
        "why_needed": "<one short sentence>"
    }
]

Rules:
- Maximum {max_tasks} tasks.
- Only propose tasks for coverage classes marked as missing.
- Do not repeat work already covered by the evidence vault.
- Do not repeat work already planned on the task board.
- Prefer narrow questions over broad ones.
- If no new tasks are needed, return [].
"""

PLANNER_HUMAN_PROMPT = """\
=== RESEARCH REQUEST ===
Ticker: {ticker}
Query: {query}
Horizon: {horizon}

=== COVERAGE STATUS ===
{coverage_summary}

=== CURRENT EVIDENCE VAULT ===
{evidence_summary}

=== EXISTING TASK BOARD ===
{existing_tasks}

=== CRITIC EVIDENCE GAPS ===
{critic_gaps}

=== INSTRUCTIONS ===
Generate up to {max_tasks} new tasks that fill missing coverage only.
Return JSON array only.
"""

PLANNER_RETRY_PROMPT ="""\
Your previous response was invalid.

Return ONLY a valid JSON array.
No prose. No markdown.

Allowed task_type values:
PRICE_DATA, TECHNICALS, NEWS_SEARCH, FUNDAMENTALS, PEER_COMPARE, EARNINGS_CHECK, FILING_SUMMARY

If no tasks are needed, return [].
"""

# ==========================================================
# NORMALIZATION / FINGERPRINTS
# ==========================================================

def _normalize_text(text: str) -> str:
    """Normalize text for matching."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

def _normalize_entity(entity: str) -> str:
    """Normalize entity names."""
    entity = entity.strip()
    if 1 <= len(entity) <= 6 and entity.replace(".", "").replace("-", "").isalnum():
        return entity.upper()
    return entity

def _question_intent_key(question: str) -> str:
    """Return an intent key for the question."""
    q = _normalize_text(question)

    patterns = [
        (r"(90 day|price trend|volume|ohlcv|price action)", "price_trend"),
        (r"(rsi|macd|bollinger|technical indicator| momtentum)", "technicals"),
        (r"(news|sentiment|coverage|catalyst|headline)", "news"),
        (r"(revenue|margin|eps|cash flow|valuation|balance sheet|fundamental)", "fundamentals"),
        (r"(peer|compare|benchmark|relative|vs)", "peer_compare"),
        (r"(earnings|q\d|quarterly|guidance|results)", "earnings_check"),
        (r"(filing|10k|10q|sec|anual report)", "filing_summary"),
    ]

    for pattern, key in patterns:
        if re.search(pattern, q):
            return key
        
    return hashlib.md5(q.encode("utf-8")).hexdigest()[:12]


def _task_fingerprint(task_type: str, entity: str, question: str) -> str:
    """Build a stable task key."""
    return f"{task_type}|{_normalize_entity(entity)}|{_question_intent_key(question)}"

# ==========================================================
# COVERAGE MODEL
# ==========================================================

COVERAGE_KEYS = {
    "PRICE_DATA",
    "TECHNICALS",
    "NEWS_SEARCH",
    "FUNDAMENTALS",
    "PEER_COMPARE",
    "EARNINGS_CHECK",
    "FILING_SUMMARY",
}

def _infer_coverage_key_from_evidence(item: EvidenceItem) -> Optional[str]:
    """Infer coverage key from evidence."""
    candidates = [
        getattr(item, "evidence_type", None),
        getattr(item, "source", None),
        getattr(item, "claim", None),
        getattr(item, "summary", None),
    ]
    
    joined = " ".join(str(x) for x in candidates if x)
    t = _normalize_text(joined)

    if any(k in t for k in ["rsi", "macd", "bollinger", "technical indicator"]):
        return "TECHNICALS"
    if any(k in t for k in ["price", "volume", "ohlcv", "trend"]):
        return "PRICE_DATA"
    if any(k in t for k in ["news", "sentiment", "catalyst", "headline", "coverage", "media"]):
        return "NEWS_SEARCH"
    if any(k in t for k in ["revenue", "margin", "eps", "cash flow", "valuation", "balance sheet", "fundamental"]):
        return "FUNDAMENTALS"
    if any(k in t for k in ["peer", "benchmark", "relative performance", "xlk", "sector compare"]):
        return "PEER_COMPARE"
    if any(k in t for k in ["earnings", "q1", "q2", "q3", "q4", "quarterly result", "guidance"]):
        return "EARNINGS_CHECK"
    if any(k in t for k in ["filing", "10-k", "10-q", "sec filling", "annual report"]):
        return "FILING_SUMMARY"
    
    return None

def _infer_coverage_key_from_task(task: ResearchTask) -> Optional[str]:
    """Infer coverage key from task type."""
    name = str(task.task_type.value if hasattr(task.task_type, "value") else task.task_type).upper()
    if name in COVERAGE_KEYS:
        return name
    return None

def _compute_coverage_state(
    evidence_vault: List[EvidenceItem],
    task_board: List[ResearchTask],
) -> Dict[str, bool]:
    """Mark covered vs missing areas."""
    coverage = {k: False for k in COVERAGE_KEYS}

    for item in evidence_vault:
        k = _infer_coverage_key_from_evidence(item)
        if k:
            coverage[k] = True

    for task in task_board:
        if task.status in (TaskStatus.DONE, TaskStatus.IN_PROGRESS, TaskStatus.PENDING):
            k = _infer_coverage_key_from_task(task)
            if k:
                coverage[k] = True

    return coverage

def _format_coverage_summary(coverage: Dict[str, bool]) -> str:
    """Format coverage as text."""
    lines = []
    for k in sorted(COVERAGE_KEYS):
        lines.append(f"- {k}: {'covered' if coverage.get(k) else 'missing'}")
    return "\n".join(lines)

# ==========================================================
# PROMPT HELPERS
# ==========================================================