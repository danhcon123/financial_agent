from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from agents.analyst import AnalystAgent
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
def _summarize_vault(evidence: List[EvidenceItem]) -> str:
    """Summarize evidence vault for prompt."""
    if not evidence:
        return "No evidence gathered yet."
    
    lines = []
    for item in evidence[:20]:
        source = getattr(item, "source", "unknown")
        evidence_type = getattr(item, "evidence_type", None)
        claim = getattr(item, "claim", "") or getattr(item, "summary", "") or ""
        confidence = getattr(item, "confidence", 0.0)
        date = getattr(item, "published_at", None) or getattr(item, "timestamp", None)

        etype = f" ({evidence_type})" if evidence_type else ""
        dstr = f" | date={date}" if date else ""
        lines.append(f"- [{source}{etype} | conf={confidence:.2f}{dstr}] {claim[:160]}")

    return "\n".join(lines)

def _summarize_existing_tasks(task_board: List[ResearchTask]) -> str:
    """Summarize existing task board for prompt."""
    if not task_board:
        return "No tasks planned yet."
    
    lines = []
    for task in task_board:
        lines.append(
            f"- [{task.task_id}] {task.task_type} | {_normalize_entity(task.entity)} | "
            f"status={task.status} | P{task.priority} | {task.question[:120]}"
        )
    return "\n".join(lines)

def _summarize_critic_gaps(critic: CriticOutput) -> str:
    """Summarize critic output for prompt."""
    lines = []
    
    gap_issues = [
        i for i in critic.critical_issues
        if str(i.issue_type) in ("EVIDENCE_GAP", "IssueType.EVIDENCE_GAP")
    ]
    for issue in gap_issues:
        lines.append(f"- severity={issue.severity} | {issue.issue}")

    for m in critic.missing_evidence or []:
        lines.append(f"- missing: {m}")
    
    return "\n".join(lines) if lines else "No evidence gaps identified by critic."

def _critic_has_evidence_gaps(critic: CriticOutput) -> bool:
    """Check if critic identified evidence gaps."""
    if critic.missing_evidence:
        return True
    for issue in critic.critical_issues:
        if str(issue.issue_type) in ("EVIDENCE_GAP", "IssueType.EVIDENCE_GAP"):
            return True
    return False


# ==========================================================
# TASK BUILDERS / VALIDATORS
# ==========================================================

def _next_task_id(existing_tasks: List[ResearchTask], cycle: int) -> str:
    """Generate next task ID."""
    existing_ids = {t.task_id for t in existing_tasks}
    n = 1
    while True:
        candidate = f"T{cycle}_{n}"
        if candidate not in existing_ids:
            return candidate
        n += 1

def _assign_dependencies(task: ResearchTask, all_tasks: List[ResearchTask]) -> List[str]:
    """Assign dependencies based on question intent."""
    by_type = {str(t.task_type.value if hasattr(t.task_type, "value") else t.task_type).upper(): t for t in all_tasks}

    current_type = str(task.task_type.value if hasattr(task.task_type, "value") else task.task_type).upper()

    if current_type == "TECHNICALS" and "PRICE_DATA" in by_type:
        return [by_type["PRICE_DATA"].task_id]
    if current_type == "NEWS_RESEARCH" and "PRICE_DATA" in by_type:
        return [by_type["PRICE_DATA"].task_id]
    return []

def _fallback_task_for_gap(
    request: ResearchRequest,
    coverage: Dict[str, bool],
    existing_tasks: List[ResearchTask],
    cycle: int,
) -> List[ResearchTask]:
    """Generate a fallback task if critical gaps are identified."""
    ticker = _normalize_entity(request.ticker or "UNKNOWN")

    candidates: List[Tuple[str, str, int, str]] = []

    if not coverage["PRICE_DATA"]:
        candidates.append((
            "PRICE_DATA",
            f"What is the 90-day price trend and volume profile for {ticker}?",
            1,
            "Baseline market data is still missing."
        ))

    if not coverage["TECHNICALS"]:
        candidates.append((
            "TECHNICALS",
            f"What are the RSI, MACD, and Bollinger Bands for {ticker} based on recent price data?",
            1,
            "Technical signal context is still missing.",
        ))
    
    if not coverage["NEWS_SEARCH"]:
        candidates.append((
            "FUNDAMENTALS",
            f"What are the lagest key fundamentals and valuation metrics for {ticker}?",
            1,
            "Core business and valuation evidence is still missing.",
        ))
    
    tasks: List[ResearchTask] = []
    for task_type, question, priority, why_needed in candidates[:1]:
        tasks = (ResearchTask(
            task_id=_next_task_id(existing_tasks + tasks, cycle),
            task_type=TaskType(task_type),
            entity=ticker,
            question=question,
            priority=priority,
            why_needed=why_needed,
            depends_on=[],
            status=TaskStatus.PENDING,
            result_evidence_ids=[],
        ))
        tasks.append(tasks)
    return tasks

def _parse_task_list(
    raw_json: Any,
    existing_tasks: List[ResearchTask],
    cycle: int,
    max_tasks: int,
    coverage: Dict[str, bool],
) -> List[ResearchTask]:
    """Parse and filter planner tasks."""
    if not isinstance(raw_json, list):
        logger.warning("[planner] Expected list, got %s", type(raw_json))
        return []
    
    valid_types = {t.value for t in TaskType}
    tasks: List[ResearchTask] = []

    existing_fingerprints: Set[str] = set()
    for t in existing_tasks:
        tt = str(t.task_type.value if hasattr(t.task_type, "value") else t.task_type).upper()
        existing_fingerprints.add(_task_fingerprint(tt, t.entity, t.question))

    for raw_task in raw_json[:max_tasks]:
        if not isinstance(raw_task, dict):
            continue

        task_type_str = str(raw_task.get("task_type", "")).upper().strip()
        if task_type_str not in valid_types:
            logger.warning("[planner] Unknown task_type '%s' — skipping", task_type_str)
            continue

        if task_type_str in coverage and coverage[task_type_str]:
            logger.info("[planner] Skipping %s because coverage already exists", task_type_str)
            continue

        entity = _normalize_entity(str(raw_task.get("entity", "")).strip())
        question = str(raw_task.get("question", "")).strip()

        if not entity or not question:
            continue

        fp = _task_fingerprint(task_type_str, entity, question)
        if fp in existing_fingerprints:
            logger.info("[planner] Duplicate task fingerprint skipped: %s", fp)
            continue

        priority_raw = raw_task.get("priority", 2)
        try:
            priority = max(1, min(3, int(priority_raw))) # clamp 1..3
        except( ValueError, TypeError):
            priority = 2

        task = ResearchTask(
            task_id=_next_task_id(existing_tasks + tasks, cycle),
            task_type=TaskType(task_type_str),
            entity=entity,
            question=question,
            priority=priority,
            why_needed=str(raw_task.get("why_needed", "")).strip() or "Gap identified by critic",
            depends_on=[], # ignore LLM deps
            status=TaskStatus.PENDING,
            result_evidence_ids=[],
        )

        tasks.append(task)
        existing_fingerprints.add(fp)
    
    for t in tasks:
        t.depends_on = _assign_dependencies(t, existing_tasks + tasks)
    
    return tasks

# ==========================================================
# INITIAL PLAN
# ==========================================================

def _is_tactical_horizon(horizon: Optional[str]) -> bool:
    """Check for short-term horizon."""
    h = _normalize_text(horizon or "")
    return any(k in h for k in ["short", "intraday", "days", "weeks", "near term", "swing"])

def _query_mentions_compare(query: Optional[str]) -> bool:
    """Check if query ask for comparison."""
    q = _normalize_text(query)
    return any(k in q for k in ["compare", "vs", "versus", "benchmark", "relative performance", "peer", "better than", "alternative"])

def _query_mentions_news(query: str) -> bool:
    """Check if query asks for news"""
    q = _normalize_text(query)
    return any(k in q for k in ["news", "headlines", "latest", "breaking", "sentiment", "catalyst", "media", "coverage", "recent", "why now"])

def build_intial_tasks(request: ResearchRequest) -> List[ResearchTask]:
    """Build initial task set."""
    tasks: List[ResearchTask] = []
    ticker = _normalize_entity(request.ticker or "UNKNOWN")
    query = request.query or ""
    
    def add_task(task_type: str, question: str, priority: int, why_needed: str) -> None:
        """Append one task."""
        tasks = ResearchTask(
            task_id=f"T0_{len(tasks)+1}",
            task_type=TaskType(task_type),
            entity=ticker,
            question=question,
            priority=priority,
            why_needed=why_needed,
            depends_on=[],
            status=TaskStatus.PENDING,
            result_evidence_ids=[],
        )
        tasks.append(tasks)

    add_task(
        "PRICE_DATA",
        f"What is the 90-day price trend and volume profile for {ticker}?",
        1,
        "Baseline price and volume evidence is required for any thesis.",
    )

    if _is_tactical_horizon(request.horizon) or _query_mentions_news(query):
        add_task(
            "NEWS_SEARCH",
            f"What recent news, sentiment, and catalysts are affecting {ticker}?",
            1,
            "Recent catalysts are important for tactical or event-driven analysis.",
        )
    
    if _is_tactical_horizon(request.horizon):
        add_task(
            "TECHNICALS",
            f"What are the RSI, MACD, and Bollinger Band signals for {ticker} based on recent price data?",
            1,
            "Technical signals matter for short-horizon timing decisions.",
        )
    else:
        add_task(
            "FUNDAMENTALS",
            f"What are the key fundamentals and valuation metrics for {ticker}?",
            1,
            "Core business quality and valuation are required for non-tactical thesis work.",
        )

    if _query_mentions_compare(query):
        add_task(
            "PEER_COMPARE",
            f"How does {ticker} compare with relevant peers on returns, valuation, and growth expectations?",
            2,
            "Relative positioning is required because the request implies comparison.",
        )

    for t in tasks:
        t.depends_on = _assign_dependencies(t, tasks)  # set deps

    logger.info("[planner] Build %d initial tasks for %s", len(tasks), ticker)
    return tasks

# ==========================================================
# PLANNER AGENT
# ==========================================================

class PlannerAgent:
    MAX_TASK_PER_CYCLE = 3
    MAX_RETRIES = 2

    def __init__(self):
        analyst = AnalystAgent()
        self.llm = analyst.llm
        logger.info("[planner] PlannerAgent initialized")

    def initial_plan(self, request: ResearchRequest) -> PlannerResult:
        """Return deterministic intial plan."""
        tasks = build_initial_tasks(request)
        return PlannerResult(
            tasks=tasks,
            planner_status="ok",
            used_llm=False,
            used_fallback=False,
            notes=["Initial plan generated without LLM"]
        )
    
    async def replan(
        self,
        request: ResearchRequest,
        state: ResearchCycleState,
        critic: CriticOutput,
        cycle: int,
    ) -> PlannerAgent:
        """Generate follow-up tasks based on critic feedback."""
        if state.research_budget <= 0:
            return PlannerResult(
                tasks=[],
                planner_status="budget_exhausted",
                used_llm=False,
                used_fallback=False,
                notes=["Research budget exhausted, no new tasks planned."]
            )
        
        if not _critic_has_evidence_gaps(critic):
            return PlannerResult(
                tasks=[],
                planner_status="no_gaps",
                used_llm=False,
                used_fallback=False,
                notes=["Critic did not identify any evidence gaps, no new tasks needed."]
            )
        
        coverage = _compute_coverage_state(state.evidence_vault, state.task_board)
        max_tasks = min(self.MAX_TASK_PER_CYCLE, state.research_budget)

        evidence_summary = _summarize_vault(state.evidence_vault)
        existing_tasks_str = _summarize_existing_tasks(state.task_board)
        critic_gaps = _summarize_critic_gaps(critic)
        coverage_summary = _format_coverage_summary(coverage)

        system_msg = SystemMessage(content=PLANNER_SYSTEM_PROMPT.format(max_tasks=max_tasks))

        base_human_msg = HumanMessage(content=PLANNER_HUMAN_PROMPT.format(
            ticker = _normalize_entity(request.ticker or "N/A"),
            query = request.query,
            horizon = request.horizon,
            coverage_summary = coverage_summary,
            evidence_summary = evidence_summary,
            existing_tasks = existing_tasks_str,
            critic_gaps = critic_gaps,
            max_tasks = max_tasks,
        ))

        parsed_json = None
        raw_output = ""

        for attempt in range(self.MAX_RETRIES):
            try:
                t0 = time.perf_counter()

                if attempt == 0:
                    messages = [system_msg, base_human_msg]
                else:
                    messages= [
                        system_msg,
                        base_human_msg,
                        HumanMessage(content=PLANNER_RETRY_PROMPT),
                    ]

                response = await self.llm.ainvoke(messages)
                raw_output = response.content
                elapsed = (time.perf_counter() - t0) * 1000

                logger.info("[planner] LLM responded in %.0fms (attempt %d)", elapsed, attempt+1)
                logger.debug("[planner] Raw output: %s", str(raw_output)[:500])

                parsed_json = extract_json(raw_output)
                if parsed_json is not None:
                    break

            except Exception as e:
                logger.error("[planner] LLM generation failed on attempt %d: %s", attempt+1, str(e))

        if parsed_json is None:
            fall_back