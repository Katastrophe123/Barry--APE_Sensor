"""Async engine / session management."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class Database:
    """Owns the engine and session factory for the process lifetime."""

    def __init__(self, url: str, *, echo: bool = False, **engine_kwargs: Any) -> None:
        self._url = url
        kwargs: dict[str, Any] = {"echo": echo, "future": True, **engine_kwargs}
        if url.startswith("sqlite"):
            # One shared in-memory connection so every session sees the same data.
            kwargs.setdefault("poolclass", StaticPool)
            kwargs.setdefault("connect_args", {"check_same_thread": False})
            kwargs.pop("pool_size", None)
            kwargs.pop("max_overflow", None)
            kwargs.pop("pool_pre_ping", None)
        self._engine: AsyncEngine = create_async_engine(url, **kwargs)
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
            autoflush=False,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> Database:
        return cls(
            settings.database_url,
            echo=settings.db_echo,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=True,
        )

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        return self._session_factory

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Transactional scope: commit on success, roll back on error."""
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def healthcheck(self) -> bool:
        """Return ``True`` when the database answers ``SELECT 1``."""
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception as exc:  # noqa: BLE001 - health endpoint must not raise
            logger.warning("database_healthcheck_failed", error=str(exc))
            return False
        return True

    async def dispose(self) -> None:
        await self._engine.dispose()
