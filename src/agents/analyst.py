from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List, Optional

from config.settings import get_settings
from config.prompts import ANALYST_SYSTEM_PROMPT, REVISION_CONTEXT_TEMPLATE
from src.models.schemas import ResearchRequest, EvidenceItem, AnalystOutput, CriticOutput
from src.models.enums import SignalType
from src.utils.logger import get_logger
from src.utils.json_parser import parse_json_safely, safe_list_extract

logger = get_logger(__name__)

class AnalystAgent:
    """Investment thesis generation agent with LangChain integration"""

    def __init__(self):
        settings = get_settings()
        self.llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.ollama_temperature,
        )

    async def draft(
        self,
        request: ResearchRequest,
        evidence: List[EvidenceItem]
    ) -> AnalystOutput:
        """Generate initial investment thesis"""
        logger.info(f"Analyst drafting thesis for {request.ticker or request.query}")

        user_prompt = self._build_draft_prompt(request, evidence)

        messages = [
            SystemMessage(content=ANALYST_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        response = await self.llm.ainvoke(messages)
        logger.debug(f"Analyst raw response length: {len(response.content)} chars")

        return self._parse_analyst_output(response.content)
    
    async def revise(
            self,
            request: ResearchRequest,
            current_thesis: AnalystOutput,
            critique_history: List[CriticOutput],
            evidence: List[EvidenceItem]
    ) -> AnalystOutput:
        """Revise thesis based on critic feedback"""
        logger.info(f"Analyst revising thesis (iteration {len(critique_history)})")
        
        user_prompt = self._build_revision_prompt(
            request, current_thesis, critique_history, evidence
        )

        messages = [
            SystemMessage(content=ANALYST_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        response = await self.llm.ainvoke(messages)
        logger.debug(f"Analyst revision response length: {len(response.content)} chars")

        return self._parse_analyst_output(response.content)
    
    def _build_draft_prompt(
            self, 
            request: ResearchRequest,
            evidence: List[EvidenceItem]
    ) -> str:
        """Format initial analysis request"""
        evidence_text = self._format_evidence(evidence)

        return f"""Research Request:
- Query: {request.query}
- Ticker: {request.ticker or "N/A"}
- Time Horizon: {request.horizon}
- Risk Profile: {request.risk_profile}
- Constraints: {", ".join(request.constraints) if request.constraints else "None"}

Available Evidence:
{evidence_text}

Task:
Generate an investment thesis addressing the research query. Respond with ONLY valid JSON (no markdown, no preamble) using this structure:

{{
    "thesis": "1-2 paragraph investment thesis",
    "bullets": ["testable point 1", "testable point 2", "..."],
    "risks": ["risk 1", "risk 2", "..."],
    "catalyst": ["catalyst 1", "catalyst 2"],
    "citations": ["E1", "E2"],
    "recommended_action": "LONG|SHORT|HOLD|NO_TRADE"
}}

Requirements:
- Cite evidence using IDs (e.g., E1, E2)
- Make bullets specific and testable
- Include 3-6 bullets, 3-6 risks, 2-5 catalysts
- Flag gaps in evidence if critical data is missing
"""
    
    def _build_revision_prompt(
        self,
        request: ResearchRequest,
        current: AnalystOutput,
        critique_history: List[CriticOutput],
        evidence: List[EvidenceItem]
    ) -> str:
        """Format revision request with critique history"""
        evidence_text = self._format_evidence(evidence)

        # Build critique history summary
        critique_summary = "\n\n".join(
            f"Iteration {i+1}:\n{critique.get_summary()}"
            for i, critique in enumerate(critique_history)
        )

        current_json = current.model_dump_json(ident=2)

        return f"""Revise your investment thesis based on critic feedback.

Original Request:
- Query: {request.query}
- Ticker: {request.ticker or "N/A"}

Current Thesis (JSON):
{current_json}

{REVISION_CONTEXT_TEMPLATE.format(critique_history=critique_summary)}

Available Evidence:
{evidence_text}

Task:
Producae a revised thesis addressing all critique issues. Respond with ONLY valid JSON using the same structure:

Requirements:
- Address all HIGH severity issues
- Soften or remove unsupported claims
- Add missing evidence citations
- Do not reintroduce previously identified problems
- Maintain thesis coherence while incorporating feedback
"""
    
    def _format_evidence(self, evidence: List[EvidenceItem]) -> str:
        """Format evidence list for prompt"""
        if not evidence:
            return "(No evidence available - flag this gap in your thesis)"
        
        lines = []
        for item in evidence:
            lines.append(
                f"-({item.id}) {item.claim}"
                f"[source: {item.source}, confidence: {item.confidence:.2f}]"
            )
        return "\n".join(lines)
    
    def _parse_analyst_output(self, content: str) -> AnalystOutput:
        """Parse LLM response into structured output with validation"""
        # Default fallback to prevent crashes
        default = {
            "thesis": "Unable to generate thesis - parsing failed",
            "bullets": ["Evidence insufficient for detailed analysis"],
            "risks": ["High uncertainty due to limited data"],
            "catalysts": [],
            "citations": [],
            "recommended_action": "HOLD"
        }

        data = parse_json_safely(content, fallback_key="thesis", default_value=default)

        # Extract and validate fields
        thesis = str(data.get("thesis", "")).strip() or default["thesis"]
        bullets = safe_list_extract(data, "bullets", max_items=8)
        risks = safe_list_extract(data, "risks", max_items=8)
        catalysts = safe_list_extract(data, "catalysts", max_items=6)
        citations= safe_list_extract(data, "citations", max_items=10)

        # Parse recommended action
        action_str = str(data.get("recommended_action", "HOLD")).upper()
        try:
            recommended_action = SignalType(action_str)
        except ValueError:
            logger.warning(f"Invalid action '{action_str}', defaulting to HOLD")
            recommended_action = SignalType.HOLD

        return AnalystOutput(
            thesis=thesis,
            bullets=bullets or default["bullets"],
            risks=risks or default["risks"],
            catalysts=catalysts,
            citations=citations,
            recommended_action=recommended_action
        )