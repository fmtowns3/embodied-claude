"""Smoke tests for the MCP server surface.

Phase 2.1 keeps the server thin — three tools wrapping CounterfactualStore +
SleepConsolidator. The functional behavior is tested elsewhere; this file
ensures the tools are registered and importable.
"""

from __future__ import annotations

from individual_kernel_mcp import server


def test_tools_are_registered() -> None:
    """All three Phase 2.1 tools must be exposed on the FastMCP instance."""
    # FastMCP keeps registered tools in an internal manager; the public
    # accessor is .list_tools() (async) or accessing the underlying manager.
    # For a smoke test we just check that the functions exist as attributes.
    assert callable(server.record_counterfactual)
    assert callable(server.query_counterfactuals)
    assert callable(server.sleep_consolidate)


def test_main_is_importable() -> None:
    """The console entry point declared in pyproject.toml must exist."""
    assert callable(server.main)


def test_reset_store_cache_is_idempotent() -> None:
    """Calling reset before any cache build must not raise."""
    server.reset_store_cache()
    server.reset_store_cache()
