# individual-kernel-mcp

Phase 2.1 of `consciousness-mcp`. See `../../README.md` for the honesty note that governs all technical documentation in this directory.

## What ships in Phase 2.1

| Module | Role |
|---|---|
| `counterfactual.py` | `CounterfactualStore` over `social-core` SQLite, typed `CounterfactualInput` / `CounterfactualRecord` |
| `sleep.py` | `SleepConsolidator` — quiet-hours-gated cron entry, writes `~/.claude/morning_briefing.json` |
| `auto_emit.py` | Pure function `extract_counterfactuals_from_plan(plan, ctx)` — produces `CounterfactualInput`s without writing |
| `server.py` (Phase 2.1 surface) | MCP server exposing `record_counterfactual`, `query_counterfactuals`, `sleep_consolidate` |

## What ships in Phase 2.3–2.5 (this package, integrated via MCP)

| Module | Role |
|---|---|
| `frame.py` | `ConsciousFrame` + `TickFrameStore` + `PredictionErrorChannels` — per-tick canonical record (FK-only, payload-poor) |
| `bottleneck.py` | `ActionBottleneck.commit_action` — one external action per tick, second attempts deferred + counterfactual'd (not yet MCP-exposed; waits for tick producer) |
| `attention_schema.py` | `AttentionSchema` + `AttentionSchemaTracker` (ring buffer cap 60, flush to SQLite) — Attention Schema Theory surface |
| `reflect.py` | `reflect_attention_schema` — on-demand modality / dwell / focus-change summary |
| `hor.py` | `HORRecord` + `HORStore` — Higher-Order Representations as EpistemicClaim specializations (Phase 1 integration) |
| `introspect.py` | `validate_hor_content` (uses agent-grammar PEG, Phase 1 first real consumer) + `introspect` composer |

MCP tools exposed by `server.py` (post-integration):

| MCP tool | Wraps |
|---|---|
| `record_tick_frame` / `get_tick_frame` / `query_tick_frames` | `TickFrameStore` (Phase 2.3) |
| `record_attention_schema` / `update_attention_from_frame` / `flush_attention_schemas` / `summarize_attention_schema` | `AttentionSchemaTracker` + `reflect_attention_schema` (Phase 2.4) |
| `record_hor` / `get_hor` / `query_hors` | `HORStore` (Phase 2.5) |
| `compose_introspection_report` | `introspect` composer (Phase 2.5) |

Internal helpers (NOT MCP-exposed by design):
- `validate_hor_content` — embedded inside `compose_introspection_report` output
- `ActionBottleneck.commit_action` — exposed in a later PR alongside the tick producer
- `extract_counterfactuals_from_plan` — Python-only; orchestrator calls it

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
