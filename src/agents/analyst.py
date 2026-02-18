from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from typing import List, Optional, Dict, Any

from src.config.settings import get_settings
from src.config.prompts import ANALYST_SYSTEM_PROMPT, REVISION_CONTEXT_TEMPLATE
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
            format="json", # Enforce JSON output
            callbacks=[StreamingStdOutCallbackHandler()],
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

        # Use retry logic
        try:
            data = await self._invoke_with_retry(messages, max_retries=2)
            return self._parse_analyst_output(data)
        except ValueError as e:
            logger.error(f"Failed to generate valid thesis: {e}")
            # Return safe fallback
            return AnalystOutput(
                thesis=f"Failed to generate thesis: {str(e)[:200]}",
                bullets=["LLM failed to produce valid output"],
                risks=["Unable to assess risks due to LLM failure"],
                catalysts=[],
                citations=[],
                recommended_action=SignalType.NO_TRADE
            )
    
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

        # Use retry logic (same as draft)
        try:
            data = await self._invoke_with_retry(messages, max_retries=2)
            return self._parse_analyst_output(data)
        except ValueError as e:
            logger.error(f"Failed to generate valid revision: {e}")
            # Return safe fallback
            return AnalystOutput(
                thesis=f"Failed to revise thesis: {str(e)[:200]}",
                bullets=["LLM failed to produce valid revision"],
                risks=["Unable to assess risks due to LLM failure"],
                catalysts=[],
                citations=[],
                recommended_action=SignalType.NO_TRADE
            )
    
    def _build_draft_prompt(
            self, 
            request: ResearchRequest,
            evidence: List[EvidenceItem]
    ) -> str:
        """Format initial analysis request"""
        evidence_text = self._format_evidence(evidence)

        prompt = f"""Research Request:
- Query: {request.query}
- Ticker: {request.ticker or "N/A"}
- Time Horizon: {request.horizon}
- Risk Profile: {request.risk_profile}
- Constraints: {", ".join(request.constraints) if request.constraints else "None"}

Available Evidence:
{evidence_text}

Your Task:
Generate an investment thesis addressing the research query. Respond with ONLY valid JSON (no markdown, no preamble) using this exact structure:

{{
    "thesis": "2-3 paragraph investment thesis",
    "bullets": [
        "Testable claim 1 with evidence citation (E1)",
        "Testable claim 2 with evidence citation (E2)",
        ...
    ],
    "risks": [
        "Risk 1 that could invalidate thesis",
        "Risk 2...",
        ...
    ],
    "catalysts": [
        "Catalyst 1 that could drive upside",
        "Catalyst 2...",
        ...
    ],
    "citations": ["E1", "E2", "E3"],
    "recommended_action": "LONG|SHORT|HOLD"
}}

Requirements:
1. Cite evidence using IDs (E1, E2, etc.) in bullets
2. Make claims testable and specific
3. Balance risks against opportunities
4. Base recommendation on evidence strength
5. Each bullet should reference at least one evidence item

Generate the thesis now.
"""
        return prompt
    
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

        current_json = current.model_dump_json(indent=2)

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
Produce a revised thesis addressing all critique issues. Respond with ONLY valid JSON using the same structure:

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

        REQUIRED_KEYS = {"thesis", "bullets", "risks", "catalysts", "citations", "recommended_actions"}

        # Check if data is a dict
        if not isinstance(data, dict):
            logger.error(f"LLM returned non-dict: {type(data)}")
            data = default
        else:
            actual_keys = set(data.keys())

            # Check for completely wrong keys (hallucination detection)
            if len(actual_keys & REQUIRED_KEYS) < 3: # Less than 3 correct keys = hallucination
                logger.error(f"LLM hallucinated wrong schema. Got keys: {actual_keys}, expected: {REQUIRED_KEYS}")
                data = default

                # Map common mistakes to correct keys
                key_mappings = {
                    "revised_thesis": "thesis",
                    "revised_bullets": "bullets",
                    "revised_risks": "risks",
                    "revised_catalysts": "catalysts",
                    "key_risks": "risks",
                    "evidence_gaps": None,
                }

                for wrong_key, correct_key in key_mappings.items():
                    if wrong_key in data:
                        if correct_key:
                            logger.warning(f"Mapping '{wrong_key}' -> '{correct_key}'")
                            data [correct_key] = data.pop(wrong_key)
                        else:
                            logger.warning(f"Ignoring invalid key '{wrong_key}'")
                            data.pop(wrong_key)
                
                # Type validation
                if "bullets" in data and not isinstance(data["bullets"], list):
                    logger.error(f"'bullets' is not a list: {type(data['bullets'])}")
                    data["bullets"] = default["bullets"]
                elif "bullets" in data and not all(isinstance(x, str) for x in data["bullets"]):
                    logger.error("'bullets' contains non-string items")
                    data["bullets"] = [str(x) for x in data["bullets"] if x]
                
                if "risks" in data and not isinstance(data["risks"], list):
                    logger.error(f"'risks' is not a list: {type(data['risks'])}")
                    data["risks"] = default["risks"]
                elif "risks" in data and not all(isinstance(x, str) for x in data["risks"]):
                    logger.error("'risks' contains non-string items")
                    data["risks"] = [str(x) for x in data["risks"] if x]
                
                if "catalysts" in data and not isinstance(data["catalysts"], list):
                    logger.error(f"'catalysts' is not a list: {type(data['catalysts'])}")
                    data["catalysts"] = default["catalysts"]
                elif "catalysts" in data and not all(isinstance(x, str) for x in data["catalysts"]):
                    logger.error("'catalysts' contains non-string items")
                    data["catalysts"] = [str(x) for x in data["catalysts"] if x]
                
                if "citations" in data and not isinstance(data["citations"], list):
                    logger.error(f"'citations' is not a list: {type(data['citations'])}")
                    data["citations"] = default["citations"]
                elif "citations" in data and not all(isinstance(x, str) for x in data["citations"]):
                    logger.error("'citations' contains non-string items")
                    data["citations"] = [str(x) for x in data["citations"] if x]

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
    
    async def _invoke_with_retry(
        self,
        messages: List,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Invoke LLM with retry on invalid JSON.

        Args:
            messages: List of messages to send
            max_retries: Number of retry attemps

        Returns:
            Parsed JSON dict
        
        Raises:
            ValueError: If all retries fail to produce valid JSON
        """
        last_content = None
        REQUIRED_KEYS = {"thesis", "bullets", "risks", "catalysts", "citations", "recommended_action"}

        for attempt in range(max_retries + 1):
            try:
                response = await self.llm.ainvoke(messages)
                content = response.content

                # Try parsing with no fallback
                data = parse_json_safely(
                    content,
                    default_value = None,
                    return_default_on_fail=False
                )

                if data is not None:
                    if isinstance(data, dict):
                        actual_keys = set(data.keys())

                        # Check if schema is correct (at least 3 required keys present)
                        if len(actual_keys & REQUIRED_KEYS) >= 3:
                            logger.debug(f"Successfully parsed JSON with valid schema on attempt {attempt + 1}")
                            return data
                        else:
                            logger.warning(f"Invalid schema. Got keys: {actual_keys}, expected: {REQUIRED_KEYS}")
                            data = None

                
                # Parsing failed, prepare retry
                last_content = content if isinstance(content, str) else str(content)
                logger.warning(f"Attemp {attempt + 1}/{max_retries + 1}: Invalid JSON or schema")

                if attempt < max_retries:
                    # Add correction message
                    messages.append(HumanMessage(content=(
                        "Your previous response was NOT valid JSON or had wrong keys.\n"
                        "Return ONLY valid JSON with these EXACT keys (no other keys allowed):\n"
                        "{\n"
                        '  "thesis": "string",\n'
                        '  "bullets": ["string1", "string2"],\n'
                        '  "risks": ["string1", "string2"],\n'
                        '  "catalysts": ["string1", "string2"],\n'
                        '  "citations": ["E1", "E2"],\n'
                        '  "recommended_action": "LONG or SHORT or HOLD"\n'
                        "}\n\n"
                        "Do NOT use keys like 'revised_thesis', 'key_risks', 'evidence_gaps'.\n"
                        "Use ONLY the 6 keys shown above.\n"
                        "No markdown, no preamble, no extra text."
                    )))
            except Exception as e:
                logger.error(f"Error during LLM invocation: {e}")
                last_content = str(e)

        # All retries exhausted
        raise ValueError(
            f"LLM failed to return valid JSON after {max_retries + 1} attempts.\n"
            f"Last output: {last_content[:300] if last_content else 'None'}"
        )