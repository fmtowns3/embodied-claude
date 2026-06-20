"""FastMCP server for individual-kernel-mcp.

Phase 2.1 surface (shipped earlier):
  - record_counterfactual
  - query_counterfactuals
  - sleep_consolidate

Phase 2.3 surface (this integration PR):
  - record_tick_frame, get_tick_frame, query_tick_frames

Phase 2.4 surface:
  - record_attention_schema, update_attention_from_frame,
    flush_attention_schemas, summarize_attention_schema

Phase 2.5 surface:
  - record_hor, get_hor, query_hors, compose_introspection_report

This package is kernel-internal. These tools have no external side effects;
boundary-mcp evaluate_action is NOT applied here. Downstream consumers
that surface introspection output to TTS / posts / DMs MUST gate at their
own boundary. ActionBottleneck.commit_action remains intentionally
unexposed until a tick producer is wired (separate PR).

See consciousness-mcp/AGENTS.md for vocabulary discipline. MCP tool
names use functional vocabulary even when the underlying Python function
name reads more naturally (e.g. summarize_attention_schema wraps
reflect_attention_schema; compose_introspection_report wraps introspect).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from mcp.server.fastmcp import FastMCP
from social_core import SocialDB

from individual_kernel_mcp.attention_schema import (
    AttentionSchemaTracker,
    Modality,
)
from individual_kernel_mcp.counterfactual import (
    CounterfactualInput,
    CounterfactualSource,
    CounterfactualStore,
)
from individual_kernel_mcp.frame import (
    ConsciousFrameInput,
    TickFrameStore,
)
from individual_kernel_mcp.hor import (
    AssertedMode,
    HORInput,
    HORStore,
    TargetKind,
)
from individual_kernel_mcp.introspect import introspect
from individual_kernel_mcp.reflect import reflect_attention_schema
from individual_kernel_mcp.sleep import SleepConsolidator

mcp = FastMCP("individual-kernel-mcp")


@dataclass(slots=True)
class IndividualKernelStores:
    db: SocialDB
    counterfactual: CounterfactualStore
    sleep: SleepConsolidator
    tick_frames: TickFrameStore
    attention_tracker: AttentionSchemaTracker
    hor_store: HORStore


@lru_cache(maxsize=1)
def _stores() -> IndividualKernelStores:
    db = SocialDB()
    counterfactual_store = CounterfactualStore(db)
    sleep_consolidator = SleepConsolidator(
        db=db,
        counterfactual_store=counterfactual_store,
    )
    return IndividualKernelStores(
        db=db,
        counterfactual=counterfactual_store,
        sleep=sleep_consolidator,
        tick_frames=TickFrameStore(db),
        attention_tracker=AttentionSchemaTracker(db=db, owner_id="self"),
        hor_store=HORStore(db),
    )


def reset_store_cache() -> None:
    """Clear cached stores so tests or env changes get a fresh shared DB."""
    if _stores.cache_info().currsize:
        _stores().db.close()
        _stores.cache_clear()


# -----------------------------------------------------------------------------
# Phase 2.1 surface
# -----------------------------------------------------------------------------


@mcp.tool()
def record_counterfactual(payload: dict[str, Any]) -> dict[str, Any]:
    """Record a rejected alternative as a typed counterfactual."""
    record = _stores().counterfactual.record(CounterfactualInput(**payload))
    return record.model_dump(mode="json")


@mcp.tool()
def query_counterfactuals(
    since: str | None = None,
    source: CounterfactualSource | None = None,
    tick_id: str | None = None,
    person_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Read recent counterfactuals (most recent first)."""
    records = _stores().counterfactual.query(
        since=since,
        source=source,
        tick_id=tick_id,
        person_id=person_id,
        limit=limit,
    )
    return [r.model_dump(mode="json") for r in records]


@mcp.tool()
def sleep_consolidate(
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the quiet-hours-gated sleep-consolidation glue."""
    result = _stores().sleep.run(force=force, dry_run=dry_run)
    return result.model_dump(mode="json")


# -----------------------------------------------------------------------------
# Phase 2.3 surface — tick frames
# -----------------------------------------------------------------------------


@mcp.tool()
def record_tick_frame(payload: dict[str, Any]) -> dict[str, Any]:
    """Record a ConsciousFrame — the per-tick canonical record.

    Payload follows ConsciousFrameInput: ts?, person_id?, ignited (bool),
    conflicted (bool), attention_target_ref?, dominant_desire?,
    winning_memory_ids (list of FKs), prediction_error {extero, intero,
    mnemonic}, affect_summary?, chosen_action_ref?, reportability ∈
    {mentionable, background_only, do_not_surface}. Returns the persisted
    frame with an assigned tick_id.
    """
    frame = _stores().tick_frames.record(ConsciousFrameInput(**payload))
    return frame.model_dump(mode="json")


@mcp.tool()
def get_tick_frame(tick_id: str) -> dict[str, Any] | None:
    """Fetch a ConsciousFrame by tick_id; null when missing."""
    frame = _stores().tick_frames.get_by_tick_id(tick_id)
    if frame is None:
        return None
    return frame.model_dump(mode="json")


@mcp.tool()
def query_tick_frames(
    since: str | None = None,
    reportability: str | None = None,
    person_id: str | None = None,
    ignited_only: bool = False,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Read recent ConsciousFrames (most recent first)."""
    frames = _stores().tick_frames.query(
        since=since,
        reportability=reportability,  # type: ignore[arg-type]
        person_id=person_id,
        ignited_only=ignited_only,
        limit=limit,
    )
    return [f.model_dump(mode="json") for f in frames]


# -----------------------------------------------------------------------------
# Phase 2.4 surface — attention schemas
# -----------------------------------------------------------------------------


@mcp.tool()
def record_attention_schema(
    focal_target_ref: str | None,
    modality: Modality,
    intensity: float,
    dwell_seconds: float = 0.0,
    predicted_next_focus: str | None = None,
    control_handle: str | None = None,
    source_tick_id: str | None = None,
    ts: str | None = None,
) -> dict[str, Any]:
    """Append an AttentionSchema snapshot to the in-memory ring buffer.

    The buffer is process-scoped (capacity 60); call flush_attention_schemas
    to persist to SQLite. modality ∈ {visual, auditory, internal, social}.
    """
    schema = _stores().attention_tracker.record(
        focal_target_ref=focal_target_ref,
        modality=modality,
        intensity=intensity,
        dwell_seconds=dwell_seconds,
        predicted_next_focus=predicted_next_focus,
        control_handle=control_handle,
        source_tick_id=source_tick_id,
        ts=ts,
    )
    return schema.model_dump(mode="json")


@mcp.tool()
def update_attention_from_frame(tick_id: str) -> dict[str, Any]:
    """Build an AttentionSchema from a previously-recorded ConsciousFrame.

    Modality is inferred from frame.attention_target_ref; intensity from
    frame.ignited (ignited→0.8, subliminal→0.3); dwell accumulates if
    focal target matches the last buffered schema.
    """
    stores = _stores()
    frame = stores.tick_frames.get_by_tick_id(tick_id)
    if frame is None:
        raise ValueError(f"no tick_frame {tick_id!r}; record_tick_frame first")
    schema = stores.attention_tracker.update_from_frame(frame)
    return schema.model_dump(mode="json")


@mcp.tool()
def flush_attention_schemas() -> dict[str, int]:
    """Persist buffered AttentionSchemas to SQLite; clears the buffer."""
    count = _stores().attention_tracker.flush()
    return {"count": count}


@mcp.tool()
def summarize_attention_schema(extra_history: int = 0) -> dict[str, Any]:
    """Build an AttentionReflection over the buffer (and optional history)."""
    stores = _stores()
    reflection = reflect_attention_schema(
        stores.attention_tracker, extra_history=extra_history
    )
    return reflection.model_dump(mode="json")


# -----------------------------------------------------------------------------
# Phase 2.5 surface — HOR + introspection
# -----------------------------------------------------------------------------


@mcp.tool()
def record_hor(payload: dict[str, Any]) -> dict[str, Any]:
    """Record a higher-order representation (HOR).

    Payload follows HORInput: ts?, owner_id (default 'self'), target_kind
    ∈ {memory, desire, action, frame, none}, target_ref?, asserted_mode ∈
    {seeing, wanting, intending, remembering, feeling, attending},
    asserted_content (non-empty string), schema_snapshot_id?,
    source_tick_id?, confidence ∈ [0,1] (default 0.6), source ∈
    {schema_readout, reflection, post_hoc, audit_subagent}.
    """
    record = _stores().hor_store.record(HORInput(**payload))
    return record.model_dump(mode="json")


@mcp.tool()
def get_hor(hor_id: str) -> dict[str, Any] | None:
    """Fetch a HORRecord by hor_id; null when missing."""
    record = _stores().hor_store.get_by_hor_id(hor_id)
    if record is None:
        return None
    return record.model_dump(mode="json")


@mcp.tool()
def query_hors(
    since: str | None = None,
    owner_id: str | None = None,
    asserted_mode: AssertedMode | None = None,
    target_kind: TargetKind | None = None,
    source_tick_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Read recent HORRecords (most recent first)."""
    records = _stores().hor_store.query(
        since=since,
        owner_id=owner_id,
        asserted_mode=asserted_mode,
        target_kind=target_kind,
        source_tick_id=source_tick_id,
        limit=limit,
    )
    return [r.model_dump(mode="json") for r in records]


@mcp.tool()
def compose_introspection_report(
    window_hours: int = 1,
    owner_id: str = "self",
) -> dict[str, Any]:
    """Compose an introspection report (Phase 2.5 read surface).

    Reads the most-recent HOR for owner_id, the current attention
    reflection from the tracker, and counterfactuals within window_hours.
    Returns IntrospectionReport with a canonical first-person statement
    + structured detail (current_hor, current_attention,
    recent_counterfactual_count, hor_validation, notes).
    """
    stores = _stores()
    report = introspect(
        hor_store=stores.hor_store,
        attention_tracker=stores.attention_tracker,
        counterfactual_store=stores.counterfactual,
        window_hours=window_hours,
        owner_id=owner_id,
    )
    return report.model_dump(mode="json")


def main() -> None:
    """Console entry point declared in pyproject.toml."""
    mcp.run()


if __name__ == "__main__":
    main()
