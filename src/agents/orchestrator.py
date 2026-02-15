import os
import uuid
import time
from typing import List, Optional
from datetime import datetime

from langchain_core.runnables import RunnableConfig

from src.config.settings import get_settings
from src.models.schemas import (
    ResearchRequest,
    EvidenceItem,
    AnalystOutput,
    CriticOutput,
    StepEvent,
    RunResult
)
from src.models.enums import StepStatus
from src.agents.analyst import AnalystAgent
from src.agents.critic import CriticAgent
from src.utils.logger import get_logger
from src.utils.file_helpers import write_json, safe_write_json
from src.tools.data_tools import fetch_and_store_price_data

logger = get_logger(__name__)

class Orchestrator:
    """
    Main workflow controller for financial research agent.

    Coordinates: Evidence gathering -> Analysis -> Critique -> Revision loop
    Features: Full observability, artifact persistence, stateless critics
    """
    
    def __init__(self, artifacts_root: Optional[str] = None):
        settings = get_settings()
        self.artifacts_root = artifacts_root or settings.artifacts_root

        # Initialize agents
        self.analyst = AnalystAgent()
        self.critic = CriticAgent()

        logger.info("Orchestrator initialized")

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
        evidence: List[EvidenceItem] = []
        # analyst_output = AnalystOutput(thesis="", recommended_action="HOLD")
        # critic_output = CriticOutput()
        analyst_output: Optional[AnalystOutput] = None
        critic_output: Optional[CriticOutput] = None
        critique_history: List[CriticOutput] = []
        iterations_completed = 0

        try:
            # ================================================================
            # INITIALIZATION
            # ================================================================
            self._record_step(trace, "init", StepStatus.START, {
                "run_id": run_id,
                "ticker": request.ticker,
                "query": request.query,
            })

            write_json(
                os.path.join(artifacts_dir, "request.json"),
                request.model_dump()
            )

            self._record_step(trace, "init", StepStatus.END)

            # ================================================================
            # EVIDENCE GATHERING (Slice 0: Mocked)
            # ================================================================
            self._record_step(trace, "fetch_evidence", StepStatus.START)
            evidence = self._fetch_evidence_stub(request)
            write_json(
                os.path.join(artifacts_dir, "evidence.json"),
                [e.model_dump() for e in evidence]
            )
            self._record_step(trace, "fetch_evidence", StepStatus.END, {
                "evidence_count": len(evidence)
            })

            # ================================================================
            # ANALYST DRAFT
            # ================================================================
            self._record_step(trace, "analyst_draft", StepStatus.START)
            analyst_output = await self.analyst.draft(request, evidence)
            write_json(
                os.path.join(artifacts_dir, "analyst_v1.json"),
                analyst_output.model_dump()
            )
            self._record_step(trace, "analyst_draft", StepStatus.END, {
                "thesis_length": len(analyst_output.thesis),
                "bullets_count": len(analyst_output.bullets),
                "action": analyst_output.recommended_action
            })
            
            # ================================================================
            # CRITIQUE INITIAL REVIEW
            # ================================================================
            self._record_step(trace, "critic_review", StepStatus.START)
            critic_output = await self.critic.review(request, analyst_output, evidence)
            critique_history.append(critic_output)
            write_json(
                os.path.join(artifacts_dir, "critic_v1.json"),
                critic_output.model_dump()
            )
            self._record_step(trace, "critic_review", StepStatus.END, {
                "assessment": critic_output.assessment,
                "issues_count": len(critic_output.critical_issues),
                "is_clean": critic_output.is_clean()
            })

            # ================================================================
            # REVISION LOOP (if needed and iterations allowed)
            # ================================================================
            while not critic_output.is_clean() and iterations_completed < request.max_iterations:
                iteration = iterations_completed + 1

                # Analyst revises based on critique history
                self._record_step(trace, f"analyst_revise_{iteration}", StepStatus.START)
                analyst_output = await self.analyst.revise(
                    request, analyst_output, critique_history, evidence
                )
                write_json(
                    os.path.join(artifacts_dir, f"analyst_v{iteration+1}.json"),
                    analyst_output.model_dump()
                )
                self._record_step(trace, f"analyst_revise_{iteration}", StepStatus.END)
                
                # Critic re-evaluates (stateless - fresh context)
                self._record_step(trace, f"critic_recheck_{iteration}", StepStatus.START)
                critic_output = await self.critic.review(request, analyst_output, evidence)
                critique_history.append(critic_output)
                write_json(
                    os.path.join(artifacts_dir, f"critic_v{iteration+1}.json"),
                    critic_output.model_dump()
                )
                self._record_step(trace, f"critic_recheck_{iteration}", StepStatus.END, {
                    "assessment": critic_output.assessment,
                    "is_clean": critic_output.is_clean()
                })

                iterations_completed = iteration

            # ================================================================
            # FINALIZATION
            # ================================================================
            write_json(
                os.path.join(artifacts_dir, "trace.json"),
                [ev.model_dump() for ev in trace]
            )

            result = RunResult(
                run_id=run_id,
                request=request,
                evidence=evidence,
                analyst_output=analyst_output,
                critic_output=critic_output,
                trace=trace,
                artifacts_dir=artifacts_dir,
                iterations_completed=iterations_completed,
                ok=True
            )

            logger.info(f"Run {run_id} completed successfully")
            logger.info(result.get_trace_summary())
            return result
        
        except Exception as e:
            logger.exception(f"Run {run_id} failed")
            
            self._record_step(trace, "run", StepStatus.ERROR, {"error": str(e)})
            safe_write_json(
                os.path.join(artifacts_dir, "trace.json"),
                [ev.model_dump() for ev in trace]
            )

            return RunResult(
                run_id=run_id,
                request=request,
                evidence=evidence,
                analyst_output=analyst_output,
                critic_output=critic_output,
                trace=trace,
                artifacts_dir=artifacts_dir,
                iterations_completed=iterations_completed,
                ok=False,
                error=str(e)
            )
        
    def _fetch_evidence_stub(self, request: ResearchRequest) -> List[EvidenceItem]:
        """
        Slice 1: Fetch REAL price data instead of stubs.

        Will be replaced with tool-based fetching in Slice 2
        """
        evidence= []
        
        # If ticker provided, fetch real price data
        if request.ticker:
            logger.info(f"Fetching real price data for {request.ticker}")

            result = fetch_and_store_price_data(request.ticker, days=90) 

            if result["success"]:
                evidence.append(EvidenceItem(
                    id="E1",
                    claim=result["evidence_claim"],
                    source="yahoo_finance",
                    timestamp=datetime.now(),
                    confidence=0.9, # High confidence - real data
                    raw=result
                ))

                logger.info(f"Successfully fetched price data for {request.ticker}")        
            else:
                logger.error(f"Failed to fetch price data: {result.get('error')}")
                # Fallback to stub
                evidence.append(EvidenceItem(
                    id="E1",
                    claim=f"Failed to fetch price data for {request.ticker}: {result.get('error')}",
                    source="error",
                    timestamp=datetime.now(),
                    confidence=0.1
                ))

        # Add placeholder for other data types (will add in later slices)
        evidence.extend([
            EvidenceItem(
                id="E2",
                claim=f"Need fundamental data (revenue, margins, guidance) for {request.ticker or request.query}.",
                source="stub_placeholder",
                timestamp=datetime.now(),
                confidence=0.2
            ),
            EvidenceItem(
                id="E3",
                claim=f"Need valuation context (peer comparisons, multiples) for {request.ticker or request.query}.",
                source="stub_placeholder",
                timestamp=datetime.now(),
                confidence=0.2
            )
        ])

        return evidence
    
    def _record_step(
        self,
        trace: List[StepEvent],
        step: str,
        status: StepStatus,
        meta = None 
    ):
        """Record execution step with timing"""
        meta = meta or {}

        # On START: store monotonic start time
        if status == StepStatus.START:
            meta["_t0"] = time.perf_counter()

        # Calculate duration for END events
        duration_ms = None
        if status == StepStatus.END and trace:
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
            meta=meta or {}
        )

        trace.append(event)
        
        # Log event
        if status == StepStatus.START:
            logger.info(f"[{step}] START {event.meta}")
        elif status == StepStatus.END:
            duration_str = f"{duration_ms:.2f}ms" if duration_ms else ""
            logger.info(f"[{step}] END {duration_str} {event.meta}")
        else:
            logger.error(f"[{step}] ERROR {event.meta}")


# ============================================================================
# CLI ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def main():
        orch = Orchestrator()
        
        request = ResearchRequest(
            query="Create an investment thesis for Apple focusing on AI opportunities",
            ticker="AAPL",
            horizon="12 months",
            risk_profile="balanced",
            constraints=["no levelrage", "must consider regulatory risks"],
            max_iterations=2 # Allow up to 2 revision cycles
        )
        
        result = await orch.run(request)
         
        print("\n" + "="*80)
        print("SLICE 0 EXECUTION COMPLETE")
        print("="*80)
        print(f"\nRun ID: {result.run_id}")
        print(f"Status: {'✅ SUCCESS' if result.ok else '❌ FAILED'}")
        print(f"Iterations: {result.iterations_completed}/{result.request.max_iterations}")
        print(f"Artifacts: {result.artifacts_dir}")
        print(f"\n{'-'*80}")
        print("ANALYST THESIS:")
        print(f"{'-'*80}")
        print(result.analyst_output.thesis)
        print(f"\nRecommended Action: {result.analyst_output.recommended_action}")
        print(f"\nKey Bullets ({len(result.analyst_output.bullets)}):")
        for i, bullet in enumerate(result.analyst_output.bullets, 1):
            print(f"  {i}. {bullet}")
        print(f"\n{'-'*80}")
        print("CRITIC ASSESSMENT:")
        print(f"{'-'*80}")
        print(f"Overall: {result.critic_output.assessment}")
        if result.critic_output.critical_issues:
            print(f"\nCritical Issues ({len(result.critic_output.critical_issues)}):")
            for issue in result.critic_output.critical_issues:
                print(f"  [{issue.severity}] {issue.issue}")
        if result.critic_output.unsupported_claims:
            print(f"\nUnsupported Claims ({len(result.critic_output.unsupported_claims)}):")
            for claim in result.critic_output.unsupported_claims:
                print(f"  - {claim}")
        print(f"\n{'='*80}\n")
    
    asyncio.run(main())