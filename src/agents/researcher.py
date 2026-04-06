"""
Researcher Agent
-------------------
Executes ResearchTasks by calling the appropriate data tool,
normalizes the raw output into typed EvidenceItems, and appends
them to the evidence vault in ResearchCycleState.

Design principles:
- No LLM calls — pure retrieval + normalization
- One handler per TaskType (matches NORMALIZER_REGISTRY pattern)
- Sets task status explicitly (IN_PROGRESS → DONE | FAILED)
- Soft-fails: a failed task never crashes the pipeline
- Returns a ResearcherResult with full execution metadata
- Budget-aware: orchestrator controls how many tasks to execute

Workflow position:
    ResearchTask (PENDING)
        ↓
    ResearcherAgent.execute_task()
        ↓
    raw dict (from tool)
        ↓
    normalizer.normalize()
        ↓
    EvidenceItem → appended to ResearchCycleState.evidence_vault
        ↓
    task.status = DONE | FAILED
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List
import time
from dataclasses import dataclass, field
from enum import Enum


from src.data.normalizer import SourceName, normalize
from src.models.enums import TaskType, TaskStatus
from src.models.schemas import (
    EvidenceItem,
    ResearchRequest,
    ResearchTask,
    ResearchCycleState,
)
from src.tools.analysis_tools import compute_technical_indicators
from src.tools.data_tools import fetch_and_store_price_data
from src.tools.news_tools import fetch_alpha_vantage_news, fetch_gdelt_news
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TaskOutcome(str, Enum):
    SUCCESS     = "success"      # handler ran + evidence added to vault
    EMPTY       = "empty"        # handler ran + no evidence (dedup or quality gate)
    NOT_IMPL    = "not_impl"     # placeholder handler — feature not built yet
    FAILED      = "failed"       # handler raised exception
    SKIPPED     = "skipped"      # dependency not met or wrong status

# ==========================================================
# RESULT OBJECT
# ==========================================================
@dataclass
class TaskResult:
    """Result of a single task execution."""
    task_id: str
    task_type: str
    status: str  # "done" | "failed" | "skipped"
    outcome: TaskOutcome = TaskOutcome.SUCCESS
    evidence_ids: List[str] = field(default_factory=list)
    error: Optional[str] = None
    duration_ms: float = 0.0
    notes: List[str] = field(default_factory=list)
    
@dataclass
class ResearcherResult:
    """Aggregate result of executing one or more tasks in a cycle."""
    tasks_attempted: int = 0
    tasks_done: int = 0
    tasks_failed: int = 0
    tasks_skipped: int = 0
    evidence_added: int = 0
    task_results: List[TaskResult] = field(default_factory=list)
    duration_ms: float = 0.0

    def any_evidence_added(self) -> bool:
        return self.evidence_added > 0
    
    def summary(self) -> str:
        return (f"ResearcherResult: {self.tasks_attempted} tasks attempted, "
                f"{self.tasks_done} done, {self.tasks_failed} failed, "
                f"{self.evidence_added} evidence items added, "
                f"duration {self.duration_ms:.2f} ms")
    
# ==========================================================
# TOOL HANDLERS
# One function per TaskType.
# Each handler returns a list of (source_name, raw_dict) tuples.
# A task can produce multiple evidence items (e.g. price data + metadata).
# ==========================================================

def _handle_price_data(
    task: ResearchTask,
    request: ResearchRequest,
    ) -> List[tuple[str, Dict[str, Any]]]:
    """
    Fetch 90-day OHLCV data from Yahoo Finance.
    Returns price data only - technicals are seperate task.
    """
    ticker = task.entity or request.ticker
    logger.info("[researcher] Fetching price data for %s", ticker)

    raw = fetch_and_store_price_data(ticker=ticker, days=90)
    return [(SourceName.YAHOO_FINANCE.value, raw)]


def _handle_technicals(
    task: ResearchTask,
    request: ResearchRequest,
    ) -> List[tuple[str, Dict[str, Any]]]:
    """
    Compute technical indicators from stored price data.
    Requires PRICE_DATA task to have run first.
    """
    ticker = task.entity or request.ticker
    logger.info("[researcher] Computing technical indicators for %s", ticker)

    raw = compute_technical_indicators(
        ticker=ticker,
        indicators=["sma_20", "rsi_14", "macd", "bbands", "trend"]
    )
    return [(SourceName.TECHNICALS_ANALYSIS.value, raw)]


def _handle_news_search(
    task: ResearchTask,
    request: ResearchRequest
) -> List[tuple[str, Dict[str, Any]]]:
    """
    Fetch news sentiment (Alpha Vantage) and global coverage (GDELT).
    Returns two evidence items per task — sentiment + coverage signal.
    """
    ticker = task.entity or request.ticker
    logger.info("[researcher] Fetching news for %s", ticker)

    results = []
    
    # Alpha Vantage - sentiment
    av_raw = fetch_alpha_vantage_news(ticker=ticker, limit=10)
    results.append((SourceName.ALPHA_VANTAGE_NEWS.value, av_raw))
    
    # GDELT - global coverage
    gdelt_raw = fetch_gdelt_news(ticker=ticker, limit=15)
    results.append((SourceName.GDELT_NEWS.value, gdelt_raw))

    return results


def _handle_fundamentals(
    task: ResearchTask,
    request: ResearchRequest
) -> List[tuple[str, Dict[str, Any]]]:
    """
    Placeholder for fundamentals retrieval (Phase 3).
    Returns a soft-fail dict so the task completes gracefully.
    """
    ticker = task.entity or request.ticker
    logger.warning(
        "[researcher] FUNDAMENTALS handler not yet implemented for %s — "
        "returning soft-fail. Wire live fundamentals fetcher in Phase 3.",
        ticker
    )
    return [(
        "fundamentals",
        {
            "success": False,
            "not_implemented": True,
            "ticker": ticker,
            "error": "Fundamentals retrieval not implemented yet. Planned for Phase 3"
        }
    )]

def _handle_peer_compare(
    task: ResearchTask,
    request: ResearchRequest
) -> List[tuple[str, Dict[str, Any]]]:
    """
    Placeholder for peer comparison retrieval (Phase 3).
    """
    ticker = task.entity or request.ticker
    logger.warning(
        "[researcher] PEER_COMPARE handler not yet implemented for %s — "
        "returning soft-fail.",
        ticker
    )
    return [(
        "peer_compare",
        {
            "success": False,
            "not_implemented": True,
            "ticker": ticker,
            "error": "Peer comparison retrieval not implemented yet. Planned for Phase 3"
        }
    )]

def _handle_earnings_check(
    task: ResearchTask,
    request: ResearchRequest
) -> List[tuple[str, Dict[str, Any]]]:
    """
    Placeholder for earnings check retrieval (Phase 3).
    """
    ticker = task.entity or request.ticker
    logger.warning(
        "[researcher] EARNINGS_CHECK handler not yet implemented for %s — "
        "returning soft-fail.",
        ticker
    )
    return [(
        "earnings_check",
        {
            "success": False,
            "not_implemented": True,
            "ticker": ticker,
            "error": "Earnings check retrieval not implemented yet. Planned for Phase 3"
        }
    )]

def _handle_filing_summary(
    task: ResearchTask,
    request: ResearchRequest
) -> List[tuple[str, Dict[str, Any]]]:
    """
    Placeholder for SEC filing summary retrieval (Phase 3).
    """
    ticker = task.entity or request.ticker
    logger.warning(
        "[researcher] FILING_SUMMARY handler not yet implemented for %s — "
        "returning soft-fail.",
        ticker
    )
    return [(
        "filing_summary",
        {
            "success": False,
            "not_implemented": True,
            "ticker": ticker,
            "error": "Filing summary retrieval not implemented yet. Planned for Phase 3"
        }
    )]

# ==========================================================
# HANDLER DISPATCH TABLE
# Adding Phase 3 source = register handler here + normalizer
# ==========================================================

TASK_HANDLER_REGISTRY = {
    TaskType.PRICE_DATA: _handle_price_data,
    TaskType.TECHNICALS: _handle_technicals,
    TaskType.NEWS_SEARCH: _handle_news_search,
    TaskType.FUNDAMENTALS: _handle_fundamentals,
    TaskType.PEER_COMPARE: _handle_peer_compare,
    TaskType.EARNINGS_CHECK: _handle_earnings_check,
    TaskType.FILING_SUMMARY: _handle_filing_summary,
}

# ==========================================================
# DEPENDENCIES CHECKER
# ==========================================================

def _dependencies_satisfied(
    task: ResearchTask,
    task_board: List[ResearchTask]
) -> bool:
    """
    Check that all tasks this task depends on are DONE
    If a dependency FAILED, we still allow execution (soft dependency).
    """
    if not task.depends_on:
        return True
    
    board_by_id = {t.task_id: t for t in task_board}
    
    for dep_id in task.depends_on:
        dep = board_by_id.get(dep_id)
        if dep is None:
            logger.warning(
                "[researcher] Dependency %s not found for task %s — proceeding anyway",
                dep_id, task.task_id
            )
            continue
        if dep.status not in (TaskStatus.DONE, TaskStatus.FAILED):
            logger.info(
                "[researcher] Dependency %s not yet complete (status=%s) — "
                "skipping task %s for now",
                dep_id, dep.status, task.task_id
            )
            return False
 
    return True

# ==========================================================
# RESEARCHER AGENT
# ==========================================================

class ResearcherAgent:
    """
    Executes ResearchTasks and populates the evidence vault.
    
    - No LLM calls - deterministic retrieval  + normalization
    - Handles one task at a time or a full pending queue
    - Sets task.status on every task it touches
    - Soft-fails: exceptions are caught, task marked FAILED, pipeline continues
    """

    def __init__(self):
        logger.info("[researcher] ResearcherAgent initialized")
        
    # ----------------------------------------------------------
    # SINGLE TASK EXECUTION
    # ----------------------------------------------------------
    
    def execute_task(
        self,
        task: ResearchTask,
        state: ResearchCycleState,
        request: ResearchRequest,
    ) -> TaskResult:
        """
        Execute a single ResearchTask.

        1. Check dependencies
        2. Mark task IN_PROGRESS
        3. Call handler -> list of (source, raw) pairs
        4. Normalize each pair -> EvidenceItem
        5. Append to vault (dedup handled by normalizer)
        6. Mark task DONE or FAILED
        7. Return TaskResult

        Args:
            task: The ResearchTask to execute (must be PENDING)
            state: Current ResearchCycleState (vault+task board)
            request: Original research request

        Returns:
            TaskResult with execution metadata
        """
        t0 = time.perf_counter()

        # -- Skip if not PENDING --
        if task.status != TaskStatus.PENDING:
            logger.info(
                "[research] Skipping task %s - status is %s",
                task.task_id, task.status
            )
            return TaskResult(
                task_id=task.task_id,
                task_type=str(task.task_type),
                status="skipped",
                outcome=TaskOutcome.SKIPPED,
                notes=[f"Task status was {task.status}, expected PENDING"],
            )
        
        # -- Check dependencies --
        if not _dependencies_satisfied(task, state.task_board):
            return TaskResult(
                task_id=task.task_id,
                task_type=str(task.task_type),
                status="skipped",
                outcome=TaskOutcome.SKIPPED,
                notes=[f"Dependencies not yet satisfied — will retry next cycle"]
            )
        
        # -- Mark IN_PROGRESS --
        task.status = TaskStatus.IN_PROGRESS
        logger.info(
            "[researcher] Executing task %s | %s | %s | %s",
            task.task_id, task.task_type, task.entity, task.question[:80]
        )

        try:
            # -- Resolve handler --
            task_type_key = task.task_type
            handler = TASK_HANDLER_REGISTRY.get(task_type_key)

            if handler is None:
                raise ValueError(
                    f"No handler registered for task_type '{task_type_key}'"
                )
            
            # -- Call handler -> raw outputs --
            raw_outputs = handler(task=task, request=request)

            # -- Normalize each output --
            evidence_ids: List[str] = []
            existing_hashes = state.vault_dedupe_hashes()
            
            for source, raw in raw_outputs:
                item = normalize(
                    source=source,
                    raw=raw,
                    ticker=task.entity or request.ticker,
                    existing_dedupe_hashes=existing_hashes,
                )

                if item is not None:
                    state.evidence_vault.append(item)
                    existing_hashes.append(item.dedupe_hash or item.id)
                    evidence_ids.append(item.id)
                    logger.info(
                        "[researcher] Added evidence %s from %s "
                        "(conf=%.2f, direction=%s)",
                        item.id, source,
                        item.confidence,
                        item.directional_impact,
                    )
                else:
                    logger.debug(
                        "[researcher] normalize() returned None for source '%s' "
                        "(duplicate or quality gate)",
                        source
                    )

            if not evidence_ids:
                # Check if any raw output was a known placeholder
                is_not_impl = any(
                    isinstance(raw, dict) and raw.get("not_implemented") is True
                    for _, raw in raw_outputs
                )

                if is_not_impl:
                    task.status = TaskStatus.FAILED
                    return TaskResult(
                        task_id=task.task_id,
                        task_type=str(task.task_type),
                        status="failed",
                        outcome=TaskOutcome.NOT_IMPL,
                        notes=["Handler not yet implemented — task marked FAILED"],
                        duration_ms=(time.perf_counter() - t0) * 1000,
                    )
                else:
                    # Handler ran successfully but normalizer filtered everything
                    # (dedup or quality gate) — this is genuinely DONE, just empty
                    task.status = TaskStatus.DONE
                    return TaskResult(
                        task_id=task.task_id,
                        task_type=str(task.task_type),
                        status="done",
                        outcome=TaskOutcome.EMPTY,
                        notes=["Handler ran but no evidence passed normalizer (dedup or quality gate)"],
                        duration_ms=(time.perf_counter() - t0) * 1000,
                    )
    
            # -- Update task --
            task.status = TaskStatus.DONE
            task.result_evidence_ids = evidence_ids

            duration_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "[researcher] Task %s DONE - %d evidence item(s) added (%.0fms)",
                task.task_id, len(evidence_ids), duration_ms
            )

            return TaskResult(
                task_id=task.task_id,
                task_type=str(task.task_type),
                status="done",
                outcome=TaskOutcome.SUCCESS,
                evidence_ids=evidence_ids,
                duration_ms=duration_ms,
                notes=[
                    f"{len(raw_outputs)} source(s) called, "
                    f"{len(evidence_ids)} item(s) added to vault"
                ],
            )
        
        except Exception as e:
            task.status = TaskStatus.FAILED
            
            duration_ms = (time.perf_counter() - t0) * 1000

            logger.error(
                "[researcher] Task %s FAILED: %s (%.0fms)",
                task.task_id, str(e), duration_ms
            )

            return TaskResult(
                task_id=task.task_id,
                task_type=str(task.task_type),
                status="failed",
                outcome=TaskOutcome.FAILED,
                error=str(e),
                duration_ms=duration_ms,
                notes=[f"Exception: {str(e)[:200]}"]
            )

# ----------------------------------------------------------
# BATCH EXECUTION — runs all PENDING tasks in priority order
# ----------------------------------------------------------

    def execute_pending(
        self,
        state: ResearchCycleState,
        request: ResearchRequest,
        max_tasks: Optional[int] = None,
    ) -> ResearcherResult:
        """
        Execute all PENDING tasks in the task board, sorted by priority.

        Args:
            state: Current ResearchCycleState
            request: Original research request
            max_tasks: Optional cap on tasks to execute this call
                        (defaults to all PENDING tasks)

        Returns:
            ResearcherResult with aggregate execution metadata
        """
        t0 = time.perf_counter()

        pending = [
            t for t in state.task_board
            if t.status == TaskStatus.PENDING
        ]

        # Sort by priority (1=highest first)
        pending.sort(key=lambda t: t.priority)

        if max_tasks is not None:
            pending = pending[:max_tasks]

        if not pending:
            logger.info("[researcher] No PENDING tasks to execute")
            return ResearcherResult(duration_ms=(time.perf_counter() - t0) * 1000)

        logger.info(
            "[researcher] Executing %d PENDING task(s) "
            "(sorted by priority)",
            len(pending)
        )            

        result = ResearcherResult()
        vault_size_before = len(state.evidence_vault)

        for task in pending:
            task_result = self.execute_task(
                task=task,
                state=state,
                request=request,
            )

            result.task_results.append(task_result)
            result.tasks_attempted += 1

            if task_result.status == "done":
                result.tasks_done += 1
            elif task_result.status == "failed":
                result.tasks_failed += 1
            elif task_result.status == "skipped":
                result.tasks_skipped += 1

        result.evidence_added = len(state.evidence_vault) - vault_size_before
        result.duration_ms = (time.perf_counter() - t0) * 1000

        logger.info("[researcher] %s", result.summary())
        return result