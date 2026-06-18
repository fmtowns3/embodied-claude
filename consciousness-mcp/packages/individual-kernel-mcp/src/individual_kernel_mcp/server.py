"""FastMCP server for individual-kernel-mcp.

Phase 2.1 surface:
  - query_counterfactuals: read-only access to the counterfactual journal
  - sleep_consolidate: trigger the quiet-hours-gated scheduler glue

See consciousness-mcp/AGENTS.md for vocabulary discipline. The MCP tool
descriptions here are functional-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from mcp.server.fastmcp import FastMCP
from social_core import SocialDB

from individual_kernel_mcp.counterfactual import (
    CounterfactualInput,
    CounterfactualSource,
    CounterfactualStore,
)
from individual_kernel_mcp.sleep import SleepConsolidator

mcp = FastMCP("individual-kernel-mcp")


@dataclass(slots=True)
class IndividualKernelStores:
    db: SocialDB
    counterfactual: CounterfactualStore
    sleep: SleepConsolidator


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
    )


def reset_store_cache() -> None:
    """Clear cached stores so tests or env changes get a fresh shared DB."""
    if _stores.cache_info().currsize:
        _stores().db.close()
        _stores.cache_clear()


@mcp.tool()
def record_counterfactual(payload: dict[str, Any]) -> dict[str, Any]:
    """Record a rejected alternative as a typed counterfactual.

    Use this when the agent commits to one action and explicitly forgoes
    another. `source` ∈ {boundary_deny, attention_lost_bid,
    deliberate_choice, policy, ignition_failed}. `evidence_type` reuses
    Phase 1's EvidenceType enum (observed/inferred/remembered/heard/
    assumed) so the row round-trips through EpistemicClaim downstream.
    """
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
    """Read recent counterfactuals (most recent first).

    Filter by ISO timestamp (`since`), originating gate (`source`),
    tick id, or person id. Returns CounterfactualRecord dicts.
    """
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
    """Run the quiet-hours-gated sleep-consolidation glue.

    Without `force`, the call is skipped unless the wrapped quiet-hours
    predicate accepts the current timestamp. With `force=True`, the
    briefing is assembled regardless of the gate. `dry_run=True` returns
    the briefing without writing morning_briefing.json to disk.
    """
    result = _stores().sleep.run(force=force, dry_run=dry_run)
    return result.model_dump(mode="json")


def main() -> None:
    """Console entry point declared in pyproject.toml."""
    mcp.run()


if __name__ == "__main__":
    main()
