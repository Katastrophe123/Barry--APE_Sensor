"""Declarative base and cross-dialect column types."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase

# JSONB on PostgreSQL (indexable, typed) and plain JSON on SQLite, which is
# what the test-suite uses so it needs no running database.
JSONType = JSON().with_variant(JSONB(), "postgresql")

TimestampType = DateTime(timezone=True)


def utcnow() -> datetime:
    """Timezone-aware UTC now, used for Python-side column defaults.

    Python-side (rather than ``server_default=func.now()``) so SQLite and
    PostgreSQL produce identically-shaped values.
    """
    return datetime.now(tz=UTC)


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    #: Shared JSON/JSONB column type, for columns declared explicitly.
    json_type = JSONType

    type_annotation_map: dict[Any, Any] = {
        dict[str, Any]: JSONType,
        list[dict[str, Any]]: JSONType,
    }
