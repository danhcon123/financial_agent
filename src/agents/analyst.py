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

        user_prompt = self.__build_draft_prompt(request, evidence)

        messages = [
            SystemMessage(content=ANALYST_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        response = await self.llm.ainvoke(messages)
        logger.debug(f"Analyst raw response length: {len(response.content)} chars")

        return self._parse_analyst_output(response.content)