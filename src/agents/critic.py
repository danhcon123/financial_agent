from langchain_ollama import ChatOllama
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List

from src.config.settings import get_settings
from src.config.prompts import CRITIC_SYSTEM_PROMPT
from src.models.schemas import ResearchRequest, EvidenceItem, AnalystOutput, CriticOutput, CriticIssue
from src.models.enums import CritiqueSeverity
from src.utils.logger import get_logger
from src.utils.json_parser import parse_json_safely, safe_list_extract

logger = get_logger(__name__)

class CriticAgent:
    """Stateless red-team evaluation agent with LangChain integration"""

    def __init__(self):
        settings = get_settings()
        # Fresh LLM instance for stateless critique
        self.llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.7,
            callbacks=[StreamingStdOutCallbackHandler()],

        )
        
    async def review(
            self,
            request: ResearchRequest,
            analyst_output: AnalystOutput,
            evidence: List[EvidenceItem]
    ) -> CriticOutput:
        """
        Evaluate analyst thesis with fresh context (stateless)

        Note: Each call is independent - no conversation history.
        """
        logger.info("Critic evaluating thesis")
        
        user_prompt = self._build_critique_prompt(request, analyst_output, evidence)

        messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        response = await self.llm.ainvoke(messages)
        logger.debug(f"Critic raw response length: {len(response.content)} chars")
        
        return self._parse_critic_output(response.content)
    
    def _build_critique_prompt(
        self,
        request: ResearchRequest,
        analyst: AnalystOutput,
        evidence: List[EvidenceItem]
    ) -> str:
        """Format critiques request with full context"""
        evidence_ids = sorted([e.id for e in evidence])
        analyst_json = analyst.model_dump_json(indent=2)

        return f"""Evaluate this investment thesis for logical soundness and evidence quality.

Research Context:
- Query: {request.query}
- Ticker: {request.ticker or "N/A"}
- Horizon: {request.horizon}
- Risk Profile: {request.risk_profile}

Analyst Thesis (JSON):
{analyst_json}

Available Evidence IDs:
{evidence_ids}

Task:
Respond with ONLY valid JSON (no markdown, no preamble) using this structure:

{{
    "assessment": "STRONG|MODERATE|WEAK",
    "critical_issues": [
        {{"issue": "description", "severity":"HIGH|MEDIUM|LOW"}},
        ...
    ],
    "missing_evidence": ["type of data needed 1", "..."],
    "unsupported_claims": ["claim 1 without citation", "..."],
    "contradictory_evidence":["point that challenges thesis", "..."],
    "recommended_revisions": ["specific improvement 1", "..."]
}}

Evaluation Criteria:
1. Are all bullets/claims backed by cited evidence (E1, E2, etc,)?
2. Does the logic follow or are there causal gaps?
3. Are risks adequately balanced against thesis?
4. Any signs of confirmation bias or cherry-picking?
5. What critical data is missing?

Be specific, constructive, and prioritize  high-severity issues.
"""
    
    def _parse_critic_output(self, content: str) -> CriticOutput:
        """Parse LLM response into structured critique"""
        # Default fallback
        default = {
            "assessment": "MODERATE",
            "critical_issues": [],
            "missing_evidence": [],
            "unsupported_claims": [],
            "contradictory_evidence": [],
            "recommended_revisions": ["Add more specific evidence citations"]
        }

        data = parse_json_safely(content, default_value=default)

        # Extract assessment
        assessment = str(data.get("assessment", "MODERATE")).upper()
        if assessment not in ["STRONG", "MODERATE", "WEAK"]:
            logger.warning(f"Invalid assessment '{assessment}', defaulting to MODERATE")
            assessment = "MODERATE"

        # Parse critical issues with severity
        critical_issues = self._parse_critical_issue(data.get("critical_issues", []))

        # Extract other fields
        missing_evidence = safe_list_extract(data, "missing_evidence", max_items=8)
        unsupported_claims = safe_list_extract(data, "unsupported_claims", max_items=10)
        contradictory_evidence = safe_list_extract(data, "contradictory_evidence", max_items=6)
        recommended_revisions = safe_list_extract(data, "recommended_revisions", max_items=8)
        
        return CriticOutput(
            assessment=assessment,
            critical_issues=critical_issues,
            missing_evidence=missing_evidence,
            unsupported_claims=unsupported_claims,
            contradictory_evidence=contradictory_evidence,
            recommended_revisions=recommended_revisions or default["recommended_revisions"]
        )
    
    def _parse_critical_issue(self, raw_issues: any) -> List[CriticIssue]:
        """Parse and validate critique issues list"""
        if not isinstance(raw_issues, list):
            return []
        
        result = []
        for item in raw_issues:
            if not isinstance(item, dict):
                continue

            issue_text = str(item.get("issue", "")).strip()
            severity_str = str(item.get("severity", "MEDIUM")).upper()
            
            if not issue_text:
                continue

            try:
                severity = CritiqueSeverity(severity_str)
            except ValueError:
                logger.warning(f"Invalid severity '{severity_str}', defaulting to MEDIUM")
                severity = CritiqueSeverity.MEDIUM
        
            result.append(CriticIssue(issue=issue_text, severity=severity))

        return result[:10]