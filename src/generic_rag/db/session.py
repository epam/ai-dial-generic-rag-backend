import functools
import inspect
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from contextvars import ContextVar

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

DbSessionMaker = async_sessionmaker()

_current_session: ContextVar[AsyncSession] = ContextVar("_current_session")

logger = logging.getLogger(__name__)


def get_current_session() -> AsyncSession:
    return _current_session.get()


@asynccontextmanager
async def _create_session() -> AsyncGenerator[AsyncSession]:
    if (session := _current_session.get(None)) is not None:
        async with session.begin_nested():
            yield session
        return

    session = DbSessionMaker()
    token = _current_session.set(session)

    try:
        async with session.begin():
            yield session
    finally:
        _current_session.reset(token)


def transaction(target):
    """Decorator for enabling automatic transaction management."""
    if inspect.isasyncgenfunction(target):

        @functools.wraps(target)
        async def wrapper(*args, **kwargs):
            async with _create_session():
                async for value in target(*args, **kwargs):
                    yield value

        return wrapper

    if inspect.iscoroutinefunction(target):

        @functools.wraps(target)
        async def wrapper(*args, **kwargs):
            async with _create_session():
                return await target(*args, **kwargs)

        return wrapper

    raise RuntimeError(f"unsupported target: {target!r}")
