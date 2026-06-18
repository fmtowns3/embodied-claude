# individual-kernel-mcp

Phase 2.1 of `consciousness-mcp`. See `../../README.md` for the honesty note that governs all technical documentation in this directory.

## What ships in Phase 2.1

| Module | Role |
|---|---|
| `counterfactual.py` | `CounterfactualStore` over `social-core` SQLite, typed `CounterfactualInput` / `CounterfactualRecord` |
| `sleep.py` | `SleepConsolidator` — quiet-hours-gated cron entry, writes `~/.claude/morning_briefing.json` |
| `auto_emit.py` | Pure function `extract_counterfactuals_from_plan(plan, ctx)` — produces `CounterfactualInput`s without writing |
| `server.py` | MCP server exposing `query_counterfactuals` and `sleep_consolidate` |

## What does NOT ship in Phase 2.1

- `tick_frames` / `ConsciousFrame` (Phase 2.3)
- `workspace.py` ignition / refractory / precision vector (Phase 2.2)
- `AttentionSchema` / `reflect_attention_schema` (Phase 2.4)
- `HORRecord` / `introspect()` (Phase 2.5)
- Action Bottleneck wrapper around `plan_response` (Phase 2.3)

Phase 2.1 deliberately stays small so the foundation (typed counterfactuals + sleep scheduler glue) ships before any tick-cadence decisions.

## Setup

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```
