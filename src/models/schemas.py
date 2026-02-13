from __future__ import annotations
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime
from .enums import SignalType, CritiqueSeverity, StepStatus

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
    """Single critique issue with severity"""
    model_config = ConfigDict(use_enum_values=True) 

    issue: str = Field(..., description="Description of the problem")
    severity: CritiqueSeverity = Field(..., description="Impact level")

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
    evidence: List[EvidenceItem] = Field(default_factory=list)
    # analyst_output: AnalystOutput
    # critic_output: CriticOutput
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