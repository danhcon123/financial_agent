"""
Orchestrator Integration Tests
-------------------------------
Tests the full research cycle loop with real API calls where needed.

Test slices:
1. Happy path - full run completes, artifacts created, vault populated
2. Router - correct routing decision for each critic output type
3. Stop conditions - each stop reason fires correctly
4. Budget enforcement — budgets decremented correctly, never go negative
5. Replan path — planner + researcher fire on evidence gaps
6. Revise path — analyst revises on reasoning issues without new research
7. Artifact persistence — all expected files written per cycle
8. Resilience — run completes even when individual sources fail

Run with:
    pytest tests/test_orchestrator_integration.py -v
    pytest tests/test_orchestrator_integration.py -v -k "test_happy_path"

Requires:
    - Ollama running locally (for analyst + critic + planner LLM calls)
    - Alpha Vantage API key in .env (or ALPHA_VANTAGE_API_KEY env var)
    - Internet access for yfinance + GDELT
"""
from __future__ import annotations
 
import asyncio
import json
import os
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch
 
import pytest
import pytest_asyncio
 
from src.agents.orchestrator import Orchestrator, _route, _should_stop, _update_gap_tracking
from src.models.enums import IssueType, CritiqueSeverity, TaskStatus, TaskType
from src.models.schemas import (
    AnalystOutput,
    CriticIssue,
    CriticOutput,
    EvidenceItem,
    ResearchCycleState,
    ResearchRequest,
    ResearchTask,
)