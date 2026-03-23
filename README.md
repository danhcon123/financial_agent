# Financial Research Agent — Project Summary

## Overview

A multi-agent LLM system that generates, critiques, and iteratively refines investment theses using real market data. Built with LangChain + Ollama (local inference), it combines price data, technical indicators, and news signals into structured, evidence-backed research reports.

---

## Architecture

```
Request → Orchestrator → Evidence Gathering → Analyst → Critic → [Revision Loop] → Final Output
```

- **Orchestrator** (`src/agents/orchestrator.py`): Coordinates the full pipeline, manages the iteration loop, and persists all artifacts. Soft-fails individual evidence sources so the pipeline completes even if a data source is unavailable.
- **Analyst** (`src/agents/analyst.py`): Generates a structured investment thesis (action, bullets, risks, catalysts, citations) using Ollama LLM. Up to 2 retries with incremental corrective prompting if LLM returns invalid JSON or wrong schema; falls back to `NO_TRADE` defaults instead of crashing.
- **Critic** (`src/agents/critic.py`): Stateless red-team evaluation (fresh context per review). Scores issues by severity (HIGH/MEDIUM/LOW) and type — `REASONING` (fixable by rewriting) vs `EVIDENCE_GAP` (requires new data). Same retry/fallback logic as the analyst.

---

## Project Structure

```
financial-research-agent/
├── src/
│   ├── agents/          # orchestrator, analyst, critic
│   ├── config/          # settings.py, prompts.py
│   ├── data/
│   │   ├── fetchers/    # yahoo_finance.py
│   │   ├── benchmark.py # sector detection + ETF mapping
│   │   ├── storage.py   # DuckDB layer (3 tables)
│   │   └── validators.py
│   ├── models/          # Pydantic schemas, enums, market_data
│   ├── tools/           # data_tools, analysis_tools, news_tools (LangChain)
│   ├── sandbox/         # analytics.py (technical indicators)
│   ├── utils/           # logger, json_parser, file_helpers
│   └── visualization/   # charts.py (Matplotlib)
├── tests/               # pytest test suite (3 slices)
├── data/
│   ├── processed/runs/  # Per-run artifacts (JSON + charts)
│   └── market_data.db   # DuckDB database
├── docs/
├── requirements.txt
└── env.example
```

---

## Implemented Features

### Data Gathering
- **Yahoo Finance** — 90-day OHLCV price history with validation (`OHLCVValidator`)
- **Alpha Vantage** — Ticker-specific news with 5-tier sentiment scoring (Bullish → Bearish); relevance filtering (skips articles with ticker relevance < 0.5); aggregate sentiment computed across all articles
- **GDELT** — Global news coverage volume, top domains and countries; 3-attempt retry with exponential backoff (10s / 20s / 30s) on rate limits; dynamic company name resolved via yfinance
- **DuckDB cache** — Three tables: `prices`, `news_articles`, `gdelt_coverage`; stable MD5-based deduplication for articles; cache-first strategy (skips API calls if data is fresh: 24h for news, 12h for GDELT coverage)

### Technical Analysis (`sandbox/analytics.py` + `tools/analysis_tools.py`)
- **Indicators**: SMA (20-period), EMA (20-period), RSI (14-period), MACD (12/26/9), Bollinger Bands (20-period, 2 std)
- **Volume analysis**: current volume vs 20-period average with signal labels (very_high / high / normal / low)
- **Trend analysis**: current price vs SMA with percentage delta and bullish/bearish/neutral signal
- **Benchmark comparison**: ticker returns vs sector-appropriate ETF, included in every indicator run
- **Sector detection + ETF selection** (`src/data/benchmark.py`): automatic sector lookup via yfinance → 17-sector GICS-to-SPDR ETF mapping (XLK, XLF, XLV, XLE, XLY, XLP, XLI, XLU, XLB, XLRE, XLC, …); falls back to SPY if detection fails
- **Sector P/E and P/B averages**: hardcoded for 17 sectors, included in benchmark evidence claim
- **Charts**: price chart (with SMA overlay) and RSI chart (with 70/50/30 reference lines) saved as PNG (`src/visualization/charts.py`)

### Agent Workflow
- **Evidence-gathering → Analyst → Critic → Revision Loop** with configurable `max_iterations`
- **Revision guard** (`only_evidence_gaps()`): halts loop early if every remaining HIGH-severity issue is an `EVIDENCE_GAP` — prevents futile rewrites when new data is required
- **LLM JSON retry**: up to 2 retries per agent call; second attempt appends corrective `HumanMessage` with explicit schema spec
- **Key remapping**: analyst parser detects common LLM hallucinated key names (e.g. `revised_thesis` → `thesis`) and remaps automatically
- **Issue type classification**: critic prompt gives explicit rules — `REASONING` if analyst could fix it with existing evidence, `EVIDENCE_GAP` if genuinely absent data

### Observability & Artifact Persistence
- **Execution trace** (`trace.json`): every step logged with START/END/ERROR status, millisecond-precision duration (`time.perf_counter()`), and a `meta` dict (ticker, article counts, error reasons, stop reason)
- **All run artifacts** saved under `data/processed/runs/{run_id}/`:
  - `request.json`, `evidence.json`, `analyst_v*.json`, `critic_v*.json`
  - `trace.json`, `*_final.json`, `*_chart.png`
- Soft failures: individual evidence source errors are caught and stored as low-confidence fallback items; the run continues

### Data Models (`src/models/`)
- **`ResearchRequest`**: query, ticker, horizon, risk_profile, constraints, max_iterations (0–5)
- **`EvidenceItem`**: id, claim, source, timestamp, confidence (0–1), raw payload
- **`AnalystOutput`**: thesis, bullets (3–8), risks, catalysts, citations, recommended_action
- **`CriticOutput`**: assessment (STRONG/MODERATE/WEAK), critical_issues, missing_evidence, unsupported_claims, contradictory_evidence, recommended_revisions; helper methods `is_clean()` and `only_evidence_gaps()`
- **`StepEvent`**: step name, status, timestamp, duration_ms, meta dict
- **`RunResult`**: full run state including trace, artifact path, iteration count, ok/error

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM inference | Ollama (deepseek-r1:8b local / qwen3.5 cloud) |
| Agent framework | LangChain, LangGraph |
| Data validation | Pydantic |
| Market data | yfinance, Alpha Vantage API, GDELT API |
| Storage | DuckDB |
| Visualization | Matplotlib, Plotly |
| Testing | pytest, pytest-asyncio |

---

## Upcoming

- Company-specific fundamental data (live earnings, EPS, P/E, P/B per ticker) — currently sector averages are hardcoded
- Enhanced analytics integrating company fundamentals into evidence
- SEC filings integration (10-K, 10-Q via SEC Edgar API)

---


## 🛠️ Installation
1. **Clone the repository**  
   ```bash
   git clone https://github.com/danhcon123/financial_agent.git
   cd financial_agent
   ```

2. **Create virtual environment and install dependencies**
   **On Windows:**
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

   **On Mac/Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```