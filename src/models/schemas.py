from __future__ import annotations
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime
from .enums import SignalType, CritiqueSeverity, StepStatus, IssueType, TaskType, TaskStatus

# ==========================================================
# REQUEST & CONTEXT
# ==========================================================

class ResearchRequest(BaseModel):
    """User's research query with context"""
    query: str = Field(..., description="Primary research question")
    ticker: Optional[str] = Field(None, description="Stock ticker symbol")
    horizon: str = Field("6-12 months", description="Investment time horizon")
    risk_profile: str = Field("balanced", description="Risk tolerance level")
    constraints: List[str] = Field(default_factory=list, description="Investment constraints")
    max_iterations: int = Field(1, ge=0, le=5, description="Max analyst-critic refinement cycles")


# ==========================================================
# EVIDENCE & DATA
# ==========================================================

class EvidenceItem(BaseModel):
    """Single piece of evidence with provenance tracking"""
    id: str = Field(..., description="Unique identifier (e.g., 'E1', 'E2')")
    claim: str = Field(..., description="Factual claim or data point")
    source: str = Field("unknown", description="Data source identifier")
    timestamp: Optional[datetime] = Field(None, description="When evidence was gathered")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Confidence score")
    raw: Optional[Dict[str, Any]] = Field(None, description="Raw data payload")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "id": "E1",
                "claim": "Revenue grew 23% YoY in Q3 2024",
                "source": "earning_transcript",
                "confidence": 0.9
            }
        }
    )

# ==========================================================
# RESEARCH TASK (Planner output)
# ==========================================================

class ResearchTask(BaseModel):
    """Structured task emitted by the Planner agent"""
    model_config = ConfigDict(use_enum_values=True)
    
    task_id: str = Field(..., description = "Unique task identifier e.g. T1, T2")
    task_type: TaskType = Field(..., description="Category of research needed")
    entity: str = Field(..., description="Primary entity e.g. ticker, company name")
    question: str = Field(..., description="Specific question this task should answer")
    priority: int = Field(1, ge=1, le=3, description="1=high, 2=medium, 3=low")
    why_needed: str = Field(..., description="Reasoning - why this fills a gap")
    depends_on: List[str] = Field(
        default_factory = list,
        description="task_ids this depends on"
    )
    status: TaskStatus = Field(TaskStatus.PENDING, description="Execution status")
    result_evidence_ids: List[str] = Field(
        default_factory=list,
        description="EvidenceItem IDs produced by this task"
    )

# ==========================================================
# ANALYST OUTPUT
# ==========================================================
class ResearchCycleState(BaseModel):
    """Full state object carried through research cycles"""
    research_question: str = Field(..., description="The original user query")
    ticker: str = Field(..., description="Primary ticker under analysis")
    
    # Task board
    task_board: List[ResearchTask] = Field(
        default_factory=list,
        description="All tasks - pending, done, failed"
    )

    # Evidence vault - append only
    evidence_vault = List[EvidenceItem] = Field(
        default_factory=list,
        description="All normalized evidence accumulated across cycles"
    )
    
    # History
    draft_history: List[AnalystOutput] = Field(
        default_factory=list,
        description="All analyst drafts in order"
    )
    critic_history: List[CriticOutput] = Field(
        default_factory=list,
        description="All critic reviews in order"
    )
    
    # Gap tracking
    open_gaps: List[str] = Field(
        default_factory=list,
        description="Gaps that have been filled"
    )
    
    # Budgets
    iteration_budget: int = Field(3, ge=0, le=5, description="Max research cycles remaining")
    research_budget: int = Field(6, ge=0, le=10, description="Max new tasks remaining across all cycles")

    # Stop signal
    stop_reason: Optional[str] = Field(None, description="Why the loop stopped")
    final_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Convenience helpers
    def vault_ids(self) -> List[str]:
        """All evidence IDs currently in vault"""
        return [e.id for e in self.evidence_vault]

    def pending_tasks(self) -> List[ResearchTask]:
        return [t for t in self.task_board if t.status == TaskStatus.PENDING]
    
    def done_tasks(Self) -> List[ResearchTask]:
        return [t for t in self.task_board if t.status == TaskStatus.DONE]
    
    def latest_draft(self) -> Optional[AnalystOutput]:
        return self.draft_history[-1] if self.draft_history else None
    
    def latest_critique(self) -> Optional[CriticOutput]:
        return self.critic_history[-1] if self.critic_history else None
    
    def evidence_count_changed(self, previous_count: int) -> bool:
        """Did the vault actually grow this cycle"""
        return len(self.evidence_vault) > previous_count
    
    def budget_exhausted(self) -> bool:
        return self.iteration_budget <= 0 or self.research_budget <= 0
    
# ==========================================================
# ANALYST OUTPUT
# ==========================================================

class AnalystOutput(BaseModel):
    """Structured investment thesis from analyst agent"""
    model_config=ConfigDict(use_enum_values=True)
    
    thesis: str = Field(..., description="Investment thesis (1-2 paragraphs)")
    bullets: List[str] = Field(
        default_factory=list,
        description="3-6 testable supporting points"
    )
    risks: List[str] = Field(
        default_factory=list,
        description="Key risk factors (3-6 items)"
    )
    catalysts: List[str] = Field(
        default_factory=list,
        description="Potential upside triggers (2-5 items)"
    )
    citations: List[str] = Field(
        default_factory=list,
        description="Evidence IDs used (e.g., ['E1', 'E2'])"
    )
    recommended_action: SignalType = Field(
        SignalType.HOLD,
        description="Investment recommendation"
    )

    @field_validator('thesis')
    @classmethod
    def thesis_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Thesis cannot be empty")
        return v.strip()

# ==========================================================
# CRITIC OUTPUT
# ==========================================================

class CriticIssue(BaseModel):
    """Single critique issue with severity and type"""
    model_config = ConfigDict(use_enum_values=True) 

    issue: str = Field(..., description="Description of the problem")
    severity: CritiqueSeverity = Field(..., description="Impact level")
    issue_type: IssueType = Field(
        IssueType.REASONING,
        description="REASONING = fixable by revision, EVIDENCE_GAP = requires new data"
    )

class CriticOutput(BaseModel):
    """Red-team evaluation from critic agent"""
    assessment: Literal["STRONG", "MODERATE", "WEAK"] = Field(
        "MODERATE",
        description="Overall thesis quality"
    )
    critical_issues: List[CriticIssue] = Field(
        default_factory = list,
        description="Identified problem with severity"
    )
    missing_evidence: List[str] = Field(
        default_factory = list,
        description="Required data not yet available"
    )
    unsupported_claims: List[str] = Field(
        default_factory = list,
        description="Statements lacking citation"
    )
    contradictory_evidence: List[str] = Field(
        default_factory = list,
        description="Data that challenges the thesis"
    )
    recommended_revisions: List[str] = Field(
        default_factory=list,
        description="Specific actionable improvements"
    )

    def is_clean(self) -> bool:
        """Check if thesis passes critique"""
        return(
            len(self.critical_issues) == 0
            and len(self.unsupported_claims) == 0
            and self.assessment == "STRONG"
        )
    
    def only_evidence_gaps(self) -> bool:
        """
        Returns True if ALL high-severity issues are evidence gaps.
        Used by orchestrator to decide whether revision would help
        """
        high_severity_issues = [
            i for i in self.critical_issues if i.severity == CritiqueSeverity.HIGH or i.severity == "HIGH"
        ]

        if not high_severity_issues:
            return False
        
        return all(
            i.issue_type == IssueType.EVIDENCE_GAP or i.issue_type == "EVIDENCE_GAP"
            for i in high_severity_issues
        )
    
    def get_summary(self) -> str:
        """Format critique for revision context"""
        lines = [f"Assessment: {self.assessment}"]

        if self.critical_issues:
            lines.append("\nCritical Issues:")
            for issue in self.critical_issues:
                lines.append(f"-[{issue.severity}] {issue.issue}")

        if self.unsupported_claims:
            lines.append("\nUnsupported Claims:")
            lines.extend(f"- {claim}" for claim in self.unsupported_claims)

        if self.recommended_revisions:
            lines.append("\nRecommended Revisions:")
            lines.extend(f"- {rev}" for rev in self.recommended_revisions)
        
        return "\n".join(lines)
    
# ==========================================================
# EXECUTION TRACE
# ==========================================================

class StepEvent(BaseModel):
    """Single execution step for observability"""
    model_config = ConfigDict(use_enum_values=True)

    step: str = Field(..., description="Step identifier")
    status: StepStatus = Field(..., description="Execution status")
    timestamp: datetime = Field(default_factory=datetime.now)
    duration_ms: Optional[float] = Field(None, description="Step duration in milliseconds")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

# ==========================================================
# FINAL OUTPUT
# ==========================================================

class RunResult(BaseModel):
    """Complete research run output with full trace"""
    run_id: str = Field(..., description="Unique run identifier")
    request: ResearchRequest
    state: Optional[ResearchCycleState] = None
    evidence: List[EvidenceItem] = Field(default_factory=list)
    analyst_output: Optional[AnalystOutput] = None
    critic_output: Optional[CriticOutput] = None    
    trace: List[StepEvent] = Field(default_factory=list)
    artifacts_dir: str = Field(..., description="Path to saved artifacts")
    iterations_completed: int = Field(0, description="Number of refinement cycles")
    ok: bool = Field(True, description="Run completed successfully")
    error: Optional[str] = Field(None, description="Error message if failed")

    def get_trace_summary(self) -> str:
        """Format execution trace for logging"""
        lines = [f"Run {self.run_id}: {'SUCCESS' if self.ok else 'FAILED'}"]
        lines.append(f"Iterations: {self.iterations_completed}/{self.request.max_iterations}")
        lines.append(f"Evidence items: {len(self.evidence)}")
        if self.critic_output:
            lines.append(f"Critic assessment: {self.critic_output.assessment}")
        return "\n".join(lines)