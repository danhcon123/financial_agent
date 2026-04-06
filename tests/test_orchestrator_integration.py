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

# ==========================================================
# FIXTURES
# ==========================================================

ARTIFACTS_ROOT = "data/test_runs"

@pytest.fixture
def artifacts_root(tmp_path) -> str:
    """Use a temp directory for all test artifacts."""
    return str(tmp_path / "test_runs")

@pytest.fixture
def basic_request() -> ResearchRequest:
    """Minimal real request - 0 iterations, just tests initial cycle."""
    return ResearchRequest(
        query="Create a short investment thesis for Apple focusing on AI.",
        ticker="AALP",
        horizon="6 months",
        risk_profile="balanced",
        constraints=["no leverage"]
        max_iterations=0 # no revision loop - just plan + research + draft + critique
    )

@pytest.fixture
def one_iteration_request() -> ResearchRequest:
    """Request with 1 revision iteration allowed."""
    return ResearchRequest(
        query="Create an investment thesis for NVIDIA focusing on AI infrastructure.",
        ticker="NVDA",
        horizon="12 months",
        risk_profile="balanced",
        constraints=["no leverage", "must consider regulatory risks"],
        max_iterations=1,
    )

@pytest.fixture
def two_iteration_request() -> ResearchRequest:
    return ResearchRequest(
        query="Evaluate Microsoft as an AI infrastructure play.",
        ticker="MSFT",
        horizon="12 months",
        risk_profile="moderate",
        constraints=["ESG compliant"],
        max_iterations=2,
    )

def make_state(
    iteration_budget: int = 2,
    research_budget: int = 4,
    vault: List[EvidenceItem] = None,
    task_board: List[ResearchTask] = None,
) -> ResearchCycleState:
    return ResearchCycleState(
        research_question="Test query",
        ticker="AAPL",
        iteration_budget=iteration_budget,
        research_budget=research_budget,
        evidence_vault=vault or [],
        task_board=task_board or []
    )

def make_evidence(n: int = 2, ticker: str = "AAPL") -> List[EvidenceItem]:
    return [
        EvidenceItem(
            id=f"E_{i}",
            claim=f"Test claim {i} for {ticker}",
            source="yahoo_finance",
            entity=ticker,
            evidence_type="PRICE_DATA",
            summary=f"Summary {i}",
            confidence=0.8,
            dedupe_hash=f"hash_{i}",
        )
        for i in range(n)
    ]

def make_clean_critic() -> CriticOutput:
    return CriticOutput(
        assessment="STRONG",
        critical_issues=[],
        missing_evidence=[],
        unsupported_claims=[],
        contradictory_evidence=[],
        recommended_revisions=[],
    )

def make_reasoning_critic() -> CriticOutput:
    return CriticOutput(
        assessment="MODERATE",
        critical_issues=[
            CriticIssue(
                issue="Thesis overclaims revenue growth without citing specific numbers",
                severity=CritiqueSeverity.HIGH,
                issue_type=IssueType.REASONING,
            )
        ],
        missing_evidence=[],
        unsupported_claims=["Revenue will grow 30% next year"],
        recommended_revisions=["Qualify growth claims with specific evidence"],
    )

def make_evidence_gap_critic() -> CriticOutput:
    return CriticOutput(
        assessment="WEAK",
        critical_issues=[
            CriticIssue(
                issue="No competitor comparison data provided",
                severity=CritiqueSeverity.HIGH,
                issue_type=IssueType.EVIDENCE_GAP,
            )
        ],
        missing_evidence=["Peer comparison: AMD, Intel performance vs NVDA"],
        unsupported_claims=[],
        recommended_revisions=["Add competitor analysis"],
    )

def make_mixed_critic() -> CriticOutput:
    """Both reasoning and evidence gap issues."""
    return CriticOutput(
        assessment="WEAK",
        critical_issues=[
            CriticIssue(
                issue="No competitor data",
                severity=CritiqueSeverity.HIGH,
                issue_type=IssueType.EVIDENCE_GAP,
            ),
            CriticIssue(
                issue="Thesis is overconfident",
                severity=CritiqueSeverity.MEDIUM,
                issue_type=IssueType.REASONING,
            ),
        ],
        missing_evidence=["AMD Q3 earnings"],
        recommended_revisions=["Reduce confidence level"],
    )

# ==========================================================
# 1. ROUTER UNIT TESTS (no API needed)
# ==========================================================

class TestRouter:
    """Test _route() decision logic in isolation."""
 
    def test_routes_stop_when_clean(self):
        state = make_state()
        critic = make_clean_critic()
        result = _route(critic, state, evidence_added_this_cycle=0)
        assert result == "stop"
        assert state.stop_reason == "thesis_approved"
 
    def test_routes_replan_on_evidence_gap(self):
        state = make_state()
        critic = make_evidence_gap_critic()
        result = _route(critic, state, evidence_added_this_cycle=2)
        assert result == "replan"
 
    def test_routes_revise_on_reasoning_only(self):
        state = make_state()
        critic = make_reasoning_critic()
        result = _route(critic, state, evidence_added_this_cycle=2)
        assert result == "revise"
 
    def test_routes_replan_when_mixed_issues(self):
        """Evidence gaps take priority over reasoning issues."""
        state = make_state()
        critic = make_mixed_critic()
        result = _route(critic, state, evidence_added_this_cycle=2)
        assert result == "replan"
 
    def test_routes_stop_when_budget_exhausted(self):
        state = make_state(iteration_budget=0, research_budget=0)
        critic = make_reasoning_critic()
        result = _route(critic, state, evidence_added_this_cycle=0)
        assert result == "stop"
        assert state.stop_reason == "budget_exhausted"