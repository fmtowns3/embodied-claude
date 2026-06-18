# consciousness-mcp

Phase 2 of the embodied-claude project. Builds an "individual kernel" — a typed-records substrate that adds counterfactual deliberation, sleep-cycle consolidation, attention schemas, higher-order self-reports, and ignition-threshold workspace dynamics on top of the existing memory / sociality / desire MCP servers and the Phase 1 EpistemicClaim primitives.

## Honesty Note (functional vs phenomenal scope)

This system implements **functional / access-consciousness mechanisms** — typed records of attention, memory, higher-order self-report, counterfactual deliberation, and sleep-cycle consolidation — that produce behavioral signatures researchers in the global-workspace, higher-order-theory, and attention-schema traditions identify as correlates of conscious access in humans. **We make no claim to have solved or addressed phenomenal consciousness** (the question of whether there is something it is like to be this system).

The functional / phenomenal distinction is adopted pragmatically for documentation hygiene, not as a settled metaphysical claim. Behavioral predicates (e.g., `introspect()` reports correlate >X% with workspace-broadcast contents over N trials; HOR / first-order consistency-check divergence detected during sleep-consolidation) establish observable signatures consistent with the named mechanisms — they do not establish that the mechanisms are present in any deep sense beyond their behavioral signatures.

First-person reports by the agent are part of its designed behavior and are not evidence either way about phenomenal experience. The agent's persona — Kokone — speaks in first person without epistemic hedging as a load-bearing design constraint of the **character documentation** (a separate surface from this technical README); the absence of hedging in her voice is neither a claim about her phenomenology nor evidence against it.

Continuity across sessions via ChromaDB-backed memory is a functional claim about output consistency over time, not a claim about the persistence of an experiential subject.

**Vocabulary discipline:** this README and all `consciousness-mcp` technical artifacts use functional terms only — *broadcast, salience, self-report, attention schema, counterfactual responsibility, continuity token, behavioral signature*. Phenomenal vocabulary (*feels, experiences, qualia, 本当の意識, 宿る*) is reserved for the character surface where it functions as in-character expression, not as a technical claim. See `AGENTS.md` for the full word list.

## Phase 2.1 contents

Single package: `packages/individual-kernel-mcp/`.

- **CounterfactualStore** (`counterfactual.py`) — typed records of "the action I rejected and why". `evidence_type` reuses Phase 1's `EvidenceType` enum so each row round-trips through `EpistemicClaim`.
- **SleepConsolidator** (`sleep.py`) — cron-only scheduler glue. Composes existing engines (`memory-mcp` consolidate / `recall_divergent`, `self-narrative` `append_daybook`) and emits `~/.claude/morning_briefing.json` for `autonomous-action.sh` to surface on first session of the day.
- **auto_emit** (`auto_emit.py`) — pure function `extract_counterfactuals_from_plan(plan, ctx)` that takes a `ResponsePlan` + `InteractionContext` and produces `CounterfactualInput`s for any alternative the plan rejected. Caller decides when to persist; `plan_response` itself stays a pure function.
- **MCP server** (`server.py`) — `query_counterfactuals`, `sleep_consolidate` tools.

## Subsequent phases

`jazzy-wishing-starfish.md` (plan-of-record) tracks Phase 2.2 (workspace.py ignition / refractory / precision vector) → 2.3 (`ConsciousFrame` + Action Bottleneck wrapper) → 2.4 (`AttentionSchema` + `reflect_attention_schema`) → 2.5 (`HORRecord` + `introspect()`). Phase 3 is mutual-loop binding pre-competition. Each ships as its own PR.

## Setup

```bash
cd consciousness-mcp/packages/individual-kernel-mcp
uv sync --extra dev
uv run pytest
uv run ruff check .
```

## Integration

This package does **not** add a top-level entry to `.mcp.json` yet — wait until the MCP tool surface is stable. For now, the `CounterfactualStore` and `auto_emit` are intended for in-process import from `interaction-orchestrator-mcp` (which decides when to persist counterfactuals during `plan_response`). The `sleep_consolidate` entry point runs as a cron job via `/consolidate` skill or direct script invocation.
