import os
import uuid
import time
from typing import List, Optional
from datetime import datetime

from langchain_core.runnables import RunnableConfig

from config.settings import get_settings
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

logger = get_logger(__name__)