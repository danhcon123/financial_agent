import pytest
from src.models.schemas import ResearchRequest
from src.agents.orchestrator import Orchestrator

@pytest.mark.asyncio
async def test_slice0_basic_workflow():
    """Test Slice 0: End-to-End orchestration with mocked data"""
    orchestrator = Orchestrator()

    request = ResearchRequest(
        query="Should I invest in Tesla given recent price volatility?",
        ticker="TSLA",
        horizon="6 months",
        risk_profile="aggressive",
        constraints=["avoid options"],
        max_iterations=1
    )

    result = await orchestrator.run(request)

    # Basic assertions
    assert result.ok is True
    assert result.run_id
    assert result.artifacts_dir
    assert len(result.evidence) > 0
    assert result.analyst_output.thesis
    assert result.analyst_output.recommended_action in ["LONG", "SHORT", "HOLD", "NO_TRADE"]
    assert result.critic_output.assessment in ["STRONG", "MODERATE", "WEAK"]

    # Verify trace
    assert len(result.trace) > 0
    assert any(e.step == "init" for e in result.trace)
    assert any(e.step == "analyst_draft" for e in result.trace)
    assert any(e.step == "critic_review" for e in result.trace)

    # Print summary
    print("\n" + "="*60)
    print("SLICE 0 TEST RESULTS")
    print("="*60)
    print(f"\nRun ID: { result.run_id }")
    print(f"Evidence items: {len(result.evidence)}")
    print(f"\nThesis ({len(result.analyst_output.thesis)} chars):")
    print(result.analyst_output.thesis[:200] + "...")
    print(f"\nAction: {result.analyst_output.recommended_action}")
    print(f"Bullets: {len(result.analyst_output.bullets)}")
    print(f"Risks: {len(result.analyst_output.risks)}")
    print(f"Catalysts: {len(result.analyst_output.catalysts)}")
    print(f"\nCritic Assessment: {result.critic_output.assessment}")
    print(f"Critical Issues: {len(result.critic_output.critical_issues)}")
    print(f"Unsupported Claims: {len(result.critic_output.unsupported_claims)}")
    print("="*60)

@pytest.mark.asyncio
async def test_slice0_revision_loop():
    """Test multi-iteration refinement"""
    orchestrator = Orchestrator()
    
    request = ResearchRequest(
        query="Evaluate Microsoft's cloud growth potential",
        ticker="MSFT",
        max_iterations=2 # Allow revisions
    )

    result = await orchestrator.run(request)

    assert result.ok is True
    assert result.iterations_completed <=2

    # If thesis wasn't clean initially, should have revision steps
    if result.iterations_completed > 0:
        assert any("revise" in e.step for e in result.trace)
        assert any("recheck" in e.step for e in result.trace)

    print(f"\nCompleted {result.iterations_completed} revision(s)")
    print(f"Final assessment: {result.critic_output.assessment}")

@pytest.mark.asyncio
async def test_slice0_json_parsing_robustness():
    """Test that JSON parsing doesnt crash the pipeline"""
    orchestrator = Orchestrator()
    
    request = ResearchRequest(
        query="Complex analysis requiring structured output",
        ticker="NVDA"
    )

    # Should complete even if LLM returns malformed JSON
    result = await orchestrator.run(request)
    assert result.ok is True
    assert result.analyst_output.thesis  # Must have *some* thesis
    assert isinstance(result.analyst_output.bullets, list)
    assert isinstance(result.critic_output.critical_issues, list)