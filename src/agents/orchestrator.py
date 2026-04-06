"""
Orchestrator v2 

Workflow:
    Request
        ↓
    Planner.initial_plan()          → task board
        ↓
    Researcher.execute_pending()    → evidence vault
        ↓
    Analyst.draft()                 → thesis v1
        ↓
    Critic.review()                 → issues
        ↓
    Router
        ├─ clean / budget exhausted → Finalize
        ├─ REASONING issues only    → Analyst.revise() → Critic.review() → Router
        └─ EVIDENCE_GAP issues      → Planner.replan()
                                        ↓
                                    Researcher.execute_pending() [new tasks only]
                                        ↓
                                    Analyst.revise() → Critic.review() → Router
"""
from __future__ import annotations

import os
import shutil
import uuid
import time
from typing import List, Optional, Any
from datetime import datetime
from pathlib import Path

from src.models.enums import StepStatus
from src.agents.analyst import AnalystAgent
from src.agents.critic import CriticAgent
from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearcherAgent, TaskOutcome
from src.config.settings import get_settings
from src.models.schemas import (
    AnalystOutput,
    CriticOutput,
    ResearchRequest,
    ResearchCycleState,
    StepEvent,
    RunResult
)
from src.utils.logger import get_logger
from src.utils.file_helpers import write_json, safe_write_json

logger = get_logger(__name__)

# ==========================================================
# STOP CONDITIONS
# ==========================================================

def _should_stop(
    state: ResearchCycleState,
    critic: CriticOutput,
    evidence_added_this_cycle: int,
) -> tuple[bool, str]:
    """
    Evaluate stop conditions after each research cycle.
    
    Returns:
        (should_stop: bool, reason: str)

    Checked in priority order:
    1. Critic says thesis is clean
    2. Both budgets exhausted
    3. No new evidence added AND only evidence gaps remain
    4. Iteration budget gone
    """
    if critic.is_clean():
        return True, "thesis_approved"
    if state.budget_exhausted():
        return True, "budget_exhausted"
    if evidence_added_this_cycle == 0 and critic.only_evidence_gaps():
        return True, "no_marginal_gain"
    if state.iteration_budget <= 0:
        return True, "max_iterations_reached"
    return False, ""

# ==========================================================
# ROUTER
# ==========================================================
def _route(
    critic: CriticOutput,
    state: ResearchCycleState,
    evidence_added_this_cycle: int,
) -> str:
    """
    Decide what happens next after a critic review.

    Returns:
        "stop" -> finalize
        "replan" -> gaps to planner -> researcher -> revise
        "revise" -> reasoning issues -> analyst directly
    """
    should_stop, reason = _should_stop(state, critic, evidence_added_this_cycle)
    if should_stop:
        logger.info("[router] STOP - reason: %s", reason)
        state.stop_reason = reason
        return "stop"
    
    has_evidence_gaps = any(
        str(i.issue_type) in ("EVIDENCE_GAP", "IssueType.EVIDENCE_GAP")
        for i in critic.critical_issues
    ) or bool(critic.missing_evidence)

    has_reasoning_issues = any(
        str(i.issue_type) in ("REASONING", "IssueType.REASONING")
        for i in critic.critical_issues
    )

    if has_evidence_gaps:
        logger.info("[router] REPLAN - evidence gaps detected")
        return "replan"
    
    if has_reasoning_issues:
        logger.info("[router] REVISE — reasoning issues only")
        return "revise"
    
    logger.info("[router] STOP — no actionable issues remaining")
    state.stop_reason = "no_actionable_issues"
    return "stop"

# ==========================================================
# GAP TRACKING HELPER
# ==========================================================

def _update_gap_tracking(
    state: ResearchCycleState,
    critic: CriticOutput
) -> int:
    """
    Update open_gaps and resolved_gaps from latest critic output.

    compare against full gap set (critical_issues + missing_evidence)
    deduplicates before extending resolved_gaps

    Returns:
        Number of gaps newly resolved this cycle
    """
    new_gap_set = set (
        [i.issue for i in critic.critical_issues]
        + (critic.missing_evidence or [])
    )
    resolved = [g for g in state.open_gaps if g not in new_gap_set]
    existing_resolved = set(state.resolved_gaps)
    new_resolved = [g for g in resolved if g not in existing_resolved]
    state.resolved_gaps.extend(new_resolved)
    
    state.open_gaps = list(new_gap_set)

    return len(new_resolved)

# ==========================================================
# ORCHESTRATOR
# ==========================================================

class Orchestrator:
    """
    Research-centric workflow controller.
 
    Coordinates:
        Planner → Researcher → Analyst → Critic → Router → [loop]
    """
    
    def __init__(self, artifacts_root: Optional[str] = None):
        settings = get_settings()
        self.artifacts_root = artifacts_root or settings.artifacts_root

        # Initialize agents
        self.analyst = AnalystAgent()
        self.critic = CriticAgent()
        self.planner    = PlannerAgent()
        self.researcher = ResearcherAgent()

        logger.info("Orchestrator initialized")
    
    # ----------------------------------------------------------
    # MAIN ENTRY POINT
    # ----------------------------------------------------------

    async def run(self, request: ResearchRequest) -> RunResult:
        """
        Execute full research workflow with observability.

        Workflow:
        1. Initialize run (generate ID, create artifacts directory)
        2. Fetch evidence (mocked in Slice 0)
        3. Analyst drafts initial thesis
        4. Critic reviews thesis
        5. If not clean and iterations remain: revise + re-critique
        6. Save all artifacts and return result
        """
        run_id = uuid.uuid4().hex[:12]
        artifacts_dir = os.path.join(self.artifacts_root, run_id)
        os.makedirs(artifacts_dir, exist_ok=True)

        trace: List[StepEvent] = []

        state = ResearchCycleState(
            research_question=request.query,
            ticker=request.ticker or "",
            iteration_budget=request.max_iterations,
            research_budget=max(request.max_iterations * 2, 4),
        )

        analyst_output: Optional[AnalystOutput] = None
        critic_output: Optional[CriticOutput] = None
        iterations_completed = 0
        evidence_added_this_cycle = 0

        try:
            # ================================================================
            # INITIALIZATION
            # ================================================================
            self._record(trace, "init", StepStatus.START, {
                "run_id": run_id,
                "ticker": request.ticker,
                "query": request.query,
                "max_iterations": request.max_iterations
            })
            write_json(
                os.path.join(artifacts_dir, "request.json"),
                request.model_dump()
            )
            self._record(trace, "init", StepStatus.END)

            # ================================================================
            # INITIAL PLAN
            # ================================================================
            self._record(trace, "initial_plan", StepStatus.START)
            planner_result = self.planner.initial_plan(request)
            state.task_board.extend(planner_result.tasks)

            self._record(trace, "initial_plan", StepStatus.END, {
                "task_created": len(planner_result.tasks),
                "task_ids": [t.task_id for t in planner_result.tasks],
            })

            # ================================================================
            # INITIAL EVIDENCE GATHERING
            # ================================================================
            self._record(trace, "initial_research", StepStatus.START)

            researcher_result = self.researcher.execute_pending(
                state=state,
                request=request
            )

            state.research_budget -= researcher_result.tasks_attempted
            evidence_added_this_cycle = researcher_result.evidence_added

            write_json(
                os.path.join(artifacts_dir, "evidence_initial.json"),
                [e.model_dump() for e in state.evidence_vault]
            )

            self._record(trace, "initial_research", StepStatus.END, {
                "tasks_attempted": researcher_result.tasks_attempted,
                "task_done": researcher_result.tasks_done,
                "tasks_failed": researcher_result.tasks_failed,
                "evidence_added": researcher_result.evidence_added,
                "research_budget_remaining": state.research_budget,
                "duration_ms": researcher_result.duration_ms,
                "not_impl_tasks": [
                    r.task_id for r in researcher_result.task_results
                    if r.outcome == TaskOutcome.NOT_IMPL
                ],
            })

            # ================================================================
            # INITIAL ANALYST DRAFT
            # ================================================================
            self._record(trace, "analyst_draft", StepStatus.START)

            analyst_output = await self.analyst.draft(request, state.evidence_vault)
            state.draft_history.append(analyst_output)
            write_json(
                os.path.join(artifacts_dir, "analyst_v1.json"),
                analyst_output.model_dump()
            )
            self._record(trace, "analyst_draft", StepStatus.END, {
                "thesis_length": len(analyst_output.thesis or ""),
                "bullets_count": len(analyst_output.bullets),
                "action": analyst_output.recommended_action
            })
            
            # ================================================================
            # CRITIQUE INITIAL REVIEW
            # ================================================================
            self._record(trace, "critic_review_1", StepStatus.START)
            critic_output = await self.critic.review(request, analyst_output, state.evidence_vault)
            state.critic_history.append(critic_output)
            state.open_gaps = (
                [i.issue for i in critic_output.critical_issues]
                + (critic_output.missing_evidence or [])
            )
            write_json(
                os.path.join(artifacts_dir, "critic_v1.json"),
                critic_output.model_dump()
            )
            self._record(trace, "critic_review_1", StepStatus.END, {
                "assessment": critic_output.assessment,
                "issues_count": len(critic_output.critical_issues),
                "missing_evidence_count": len(critic_output.missing_evidence or []),
                "is_clean": critic_output.is_clean(),
                "only_evidence_gaps": critic_output.only_evidence_gaps(),
                "open_gaps": len(state.open_gaps)
            })

            # ============================================================
            # RESEARCH CYCLE LOOP
            # ============================================================

            while True:
                route = _route(critic_output, state, evidence_added_this_cycle)
                
                if route == "stop":
                    break

                iterations_completed += 1
                cycle   = iterations_completed
                state.iteration_budget -= 1
                evidence_added_this_cycle = 0

                # ── REPLAN + ENRICH ──────────────────────────────────────
                if route == "replan":
                    self._record(
                        trace, f"replan_{cycle}", StepStatus.START, {
                            "cycle": cycle,
                            "research_budget": state.research_budget,
                        }
                    )

                    replan_result = await self.planner.replan(
                        request=request,
                        state=state,
                        critic=critic_output,
                        cycle=cycle,
                    )

                    self._record(trace,  f"replan_{cycle}", StepStatus.END, {
                        "new_tasks": len(replan_result.tasks),
                        "planner_status": replan_result.planner_status,
                        "used_fallback": replan_result.used_fallback,
                        "research_budget_remaining": state.research_budget,
                    })

                    # Not stop blindly if planner yields no tasks
                    if not replan_result.tasks:
                        has_reasoning = any(
                            str(i.issue_type) in ("REASONING", "IssueType.REASONING")
                            for i in critic_output.critical_issues
                        )
                        if has_reasoning:
                            logger.info(
                                "[orchestrator] Planner returned no tasks - "
                                "falling back to revise (reasoning issues remain)"
                            )
                            route = "revise"
                        else:
                            logger.info(
                                "[orchestrator] Planner returned no tasks and "
                                "no reasoning issues - stopping (status=%s)",
                                replan_result.planner_status
                            )
                            state.stop_reason = replan_result.planner_status
                            break
                    else:
                        # add new tasks then cap execute to their count
                        state.task_board.extend(replan_result.tasks)

                        self._record(
                            trace, f"enrich_{cycle}", StepStatus.START
                        )

                        vault_before = len(state.evidence_vault)

                        enrich_result =  self.researcher.execute_pending(
                            state=state,
                            request=request,
                            # pnly execute newly added tasks
                            max_tasks=len(replan_result.tasks)
                        )

                        # charge budget on execution
                        state.research_budget -= enrich_result.tasks_attempted
                        evidence_added_this_cycle = (
                            len(state.evidence_vault) - vault_before
                        )

                        write_json(
                            os.path.join(
                                artifacts_dir,
                                f"evidence_cycle_{cycle}.json"
                            ),
                            [e.model_dump() for e in state.evidence_vault]
                        )

                        self._record(trace, f"enrich_{cycle}", StepStatus.END, {
                            "task_attempted": enrich_result.tasks_attempted,
                            "tasks_done": enrich_result.tasks_done,
                            "evidence_added": evidence_added_this_cycle,
                            "vault_total": len(state.evidence_vault),
                            "research_budget_remaining": state.research_budget,
                            "not_impl_tasks": [
                                r.task_id for r in enrich_result.task_results
                                if r.outcome == TaskOutcome.NOT_IMPL
                            ]
                        })

                        # stop only if evidence gaps are the core blocker
                        if evidence_added_this_cycle == 0:
                            if critic_output.only_evidence_gaps():
                                logger.info(
                                    "[orchestrator] Enrichment added no evidence "
                                    "and only evidence gaps remain - stopping"
                                )
                                state.stop_reason = "no_marginal_gain"
                                break
                            else:
                                logger.info(
                                    "[orchestrator] Enrichment added no new evidence "
                                    "but reasoning issues remain - continuing to revise"
                                )

                # ── ANALYST REVISE ───────────────────────────────────────
                # Run after both "replan" and "revise" routes
                self._record(
                    trace, f"analyst_revise_{cycle}", StepStatus.START
                )

                analyst_output = await self.analyst.revise(
                    request=request,
                    analyst_output=analyst_output,
                    critic_history=state.critic_history,
                    evidence=state.evidence_vault,
                )
                state.draft_history.append(analyst_output)

                write_json(
                    os.path.join(
                        artifacts_dir, f"analyst_v{cycle + 1}.json"
                    ),
                    analyst_output.model_dump()
                )

                self._record(trace, f"analyst_revise_{cycle}", StepStatus.END, {
                    "thesis_length": len(analyst_output.thesis or ""),
                    "action": analyst_output.recommended_action,
                    "vault_size": len(state.evidence_vault)
                })

                # ── CRITIC RECHECK ───────────────────────────────────────
                self._record(
                    trace, f"critic_recheck_{cycle}", StepStatus.START
                )

                critic_output = await self.critic.review(
                    request=request,
                    analyst_output=analyst_output,
                    evidence=state.evidence_vault,
                )
                state.critic_history.append(critic_output)

                # correct gap tracking
                gaps_resolved=_update_gap_tracking(state, critic_output)

                write_json(
                    os.path.join(
                        artifacts_dir, f"critic_v{cycle + 1}.json"
                    ),
                    critic_output.model_dump()
                ) 

                self._record(trace, f"critic_recheck_{cycle}", StepStatus.END, {
                    "assessment": critic_output.assessment,
                    "is_clean": critic_output.is_clean(),
                    "issue_count": len(critic_output.critical_issues),
                    "gaps_resolved_this_cycle": gaps_resolved,
                    "open_gaps": len(state.open_gaps),
                    "resolved_gaps_total": len(state.resolved_gaps),
                })

            # ================================================================
            # FINALIZATION
            # ================================================================
            state.stop_reason = state.stop_reason or "loop_exited"

            # Generate charts and save all artifacts
            self._finalize_run(
                run_id=run_id,
                request=request,
                state=state,                
                analyst_output=analyst_output,
                critic_output=critic_output,
                artifacts_dir=artifacts_dir,                
            )

            # Save execution trace
            write_json(
                os.path.join(artifacts_dir, "trace.json"),
                [ev.model_dump() for ev in trace]
            )

            write_json(
                os.path.join(artifacts_dir, "cycle_state.json"),
                {
                    "stop_reason": state.stop_reason,
                    "iterations_completed": iterations_completed,
                    "evidence_vault_size": len(state.evidence_vault),
                    "task_total": len(state.task_board),
                    "task_done": len(state.done_tasks()),
                    "open_gaps": state.open_gaps,
                    "resolved_gaps": state.resolved_gaps,
                    "iteration_budget_remaining": state.iteration_budget,
                    "research_budget_remaining": state.research_budget,
                }
            )

            result = RunResult(
                run_id=run_id,
                request=request,
                state=state,
                evidence=state.evidence_vault,
                analyst_output=analyst_output,
                critic_output=critic_output,
                trace=trace,
                artifacts_dir=artifacts_dir,
                iterations_completed=iterations_completed,
                ok=True
            )
            logger.info("[orchestrator] Run %s completed", run_id)
            logger.info("[orchestrator] Stop reason: %s", state.stop_reason)
            logger.info(result.get_trace_summary())
            return result
        
        except Exception as e:
            logger.exception("[orchestrator] Run %s failed: %s", run_id, e)
 
            self._record(trace, "run", StepStatus.ERROR, {"error": str(e)})
            safe_write_json(
                os.path.join(artifacts_dir, "trace.json"),
                [ev.model_dump() for ev in trace]
            )
 
            return RunResult(
                run_id=run_id,
                request=request,
                state=state,
                evidence=state.evidence_vault,
                analyst_output=analyst_output,
                critic_output=critic_output,
                trace=trace,
                artifacts_dir=artifacts_dir,
                iterations_completed=iterations_completed,
                ok=False,
                error=str(e),
            )
    
    # ----------------------------------------------------------
    # FINALIZATION
    # ----------------------------------------------------------
    
    def _finalize_run(
        self,
        run_id: str,
        request: ResearchRequest,
        state: ResearchCycleState,
        analyst_output: Optional[AnalystOutput],
        critic_output: Optional[CriticOutput],
        artifacts_dir: str,
    ) -> None:
        """
        Save all artifacts and generate final charts

        Args:
            run_id: Unique run identifier
            request: Original research request
            evidence: All evidence gathered
            analyst_output: Final analyst thesis (if exists)
            critic_output: Final critic evaluation (if exists)
            iterations: Number of revision iterations completed
        """
        if analyst_output:
            write_json(
                os.path.join(artifacts_dir, "analyst_final.json"),
                analyst_output.model_dump()
            )

        if critic_output:
            write_json(
                os.path.join(artifacts_dir, "critic_final.json"),
                critic_output.model_dump()
            )

        # Generate charts (slice 2)
        if request.ticker and analyst_output:
            self._generate_charts(request.ticker, Path(artifacts_dir))
        
        logger.info("[orchestrator] Artifacts saved: %s", artifacts_dir)

    def _generate_charts(self, ticker: str, artifacts_path: Path) -> None:
        try:
            from src.tools.analysis_tools import generate_stock_chart
            for chart_type, indicators in [
                ("price", ["sma_20"]),
                ("rsi", [])
            ]:
                result = generate_stock_chart(
                    ticker=ticker,
                    chart_type=chart_type,
                    include_indicators=indicators
                )
                if result.get("success"):
                    src=Path(result["chart_path"])
                    shutil.copy(src, artifacts_path / src.name)
                    logger.info("[orchestrator] Chart saved: %s", src.name)
                else:
                    logger.warning(
                        "[orchestrator] Chart failed (%s): %s",
                        chart_type, result.get("error")
                    )
        except Exception as e:
            logger.error("[orchestrator] Chart error: %s", e)
     
    # ----------------------------------------------------------
    # TRACE HELPER
    # ----------------------------------------------------------

    def _save_json(self, filepath: Path, data: Any) -> None:
        """Helper to save JSON data"""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
            
    def _record(
        self,
        trace: List[StepEvent],
        step: str,
        status: StepStatus,
        meta: Optional[dict] = None,
    ) -> None:
        """Record execution step with timing"""
        meta = meta or {}

        # On START: store monotonic start time
        if status == StepStatus.START:
            meta["_t0"] = time.perf_counter()

        # Calculate duration for END events
        duration_ms = None
        if status in (StepStatus.END, StepStatus.ERROR):
            # Find matching START event
            for event in reversed(trace):
                if event.step == step and event.status == StepStatus.START:
                    t0 = event.meta.get("_t0")
                    if t0 is not None:
                        duration_ms = (time.perf_counter() - t0) * 1000
                    break

        event = StepEvent(
            step=step,
            status=status,
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            meta=meta
        )
        trace.append(event)
        
        # Log event
        if status == StepStatus.START:
            logger.info(f"[trace] %s START %s", step, meta)
        elif status == StepStatus.END:
            duration_str = f"{duration_ms:.0f}ms" if duration_ms else ""
            logger.info(f"[trace] %s END %s %s", step, duration_str, meta)
        else:
            logger.error("[trace] %s ERROR %s", step, meta)

# ============================================================================
# CLI ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def main():
        orch = Orchestrator()
        
        request = ResearchRequest(
            query="Create an investment thesis for NVIDIA focusing on AI infrastructure demand",
            ticker="NVDA",
            horizon="12 months",
            risk_profile="balanced",
            constraints=["no levelrage", "must consider regulatory risks"],
            max_iterations=2 # Allow up to 2 revision cycles
        )
        
        result = await orch.run(request)
         
        print("\n" + "="*80)
        print("ORCHESTRATOR v2 — RUN COMPLETE")
        print("="*80)
        print(f"\nRun ID: {result.run_id}")
        print(f"Status: {'SUCCESS' if result.ok else 'FAILED'}")
        print(f"Iterations: {result.iterations_completed}")
        print(f"Evidence:    {len(result.evidence)} items in vault")
        print(f"Artifacts:   {result.artifacts_dir}")

        # Show why the loop stopped
        if result.state:
            print(f"Stop reason: {result.state.stop_reason}")
            print(f"Open gaps:   {len(result.state.open_gaps)}")
            print(f"Resolved:    {len(result.state.resolved_gaps)}")
            print(
                f"Budgets:     iter={result.state.iteration_budget} remaining, "
                f"research={result.state.research_budget} remaining"
            )
            print(f"\nTask Board ({len(result.state.task_board)} tasks):")
            for t in result.state.task_board:
                print(
                    f"  [{t.task_id}] {t.task_type} | {t.status} | "
                    f"{t.question[:60]}"
                )
        
        if result.analyst_output:
            print(f"\n{'─' * 80}")
            print("FINAL THESIS:")
            print(f"{'─' * 80}")
            print(result.analyst_output.thesis)
            print(f"\nAction: {result.analyst_output.recommended_action}")
            print(f"\nBullets ({len(result.analyst_output.bullets)}):")
            for i, b in enumerate(result.analyst_output.bullets, 1):
                print(f"  {i}. {b}")
        
        if result.critic_output:
            print(f"\n{'─' * 80}")
            print("FINAL CRITIQUE:")
            print(f"{'─' * 80}")
            print(f"Assessment: {result.critic_output.assessment}")
            if result.critic_output.critical_issues:
                print(f"Issues ({len(result.critic_output.critical_issues)}):")
                for issue in result.critic_output.critical_issues:
                    print(f"  [{issue.severity}] {issue.issue}")
            else:
                print("No critical issues — thesis approved.")
        
        print(f"\n{'=' * 80}\n")
    
    asyncio.run(main())