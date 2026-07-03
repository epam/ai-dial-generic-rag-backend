import logging
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from typing import Any

import click
import sqlalchemy
import sqlparse
from sqlalchemy import Connection, Dialect, Executable, event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import ConnectionPoolEntry
from sqlalchemy.util import immutabledict
from yoyo import get_backend, read_migrations

import generic_rag.db.migrations
from generic_rag.app.settings import DatabaseConfig
from generic_rag.db.auth import MsiTokenProvider

logger = logging.getLogger(__name__)


def _apply_migrations(url: sqlalchemy.engine.URL):
    migration_source = generic_rag.db.migrations.__path__[0]
    migrations = read_migrations(migration_source)
    logger.info(f"Loaded {len(migrations)} migration(s)")

    # yoyo migrations does not have backend for asyncpg
    if url.drivername == "postgresql+asyncpg":
        url = url.set(drivername="postgresql+psycopg")

    backend = get_backend(url.render_as_string(hide_password=False))
    backend.apply_migrations(migrations=backend.to_apply(migrations))
    logger.info("All migrations applied")


async def get_engine(config: DatabaseConfig, exit_stack: AsyncExitStack) -> AsyncEngine:
    """Get database engine."""
    engine = create_async_engine(config.get_url(), pool_pre_ping=True)
    exit_stack.push_async_callback(engine.dispose)
    url = engine.url

    if config.msi_enabled:
        token_provider = await exit_stack.enter_async_context(MsiTokenProvider())

        # noinspection PyUnusedLocal
        @event.listens_for(engine.sync_engine, "do_connect")
        def do_connect(dialect: Dialect, conn_rec: ConnectionPoolEntry, cargs: tuple, cparams: dict):
            cparams["password"] = token_provider.token

        url = engine.url.set(password=token_provider.token)

    # noinspection PyUnusedLocal
    @event.listens_for(engine.sync_engine, "before_execute")
    def before_execute(
        conn: Connection,
        executable: Executable,
        multiparams: Sequence[Mapping[str, Any]],
        params: Mapping[str, Any],
        execution_options: immutabledict[str, Any],
    ):
        if not logger.isEnabledFor(logging.DEBUG):
            return

        query = sqlparse.format(str(executable).strip(), keyword_case="upper", reindent=True)
        logger.debug(
            f"executing SQL query: \n{query}",
            extra={"color_message": f"executing SQL query: \n{click.style(query, fg='cyan')}"},
        )

    async with engine.connect():
        logger.info(f"Connected to DB at: {engine.url}")

    _apply_migrations(url)

    return engine
