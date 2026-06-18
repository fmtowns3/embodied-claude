"""Shared fixtures for individual-kernel-mcp tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from social_core.db import SocialDB


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "social.db"


@pytest.fixture
def social_db(temp_db_path: Path) -> Iterator[SocialDB]:
    db = SocialDB(temp_db_path)
    db.connect()
    try:
        yield db
    finally:
        db.close()
