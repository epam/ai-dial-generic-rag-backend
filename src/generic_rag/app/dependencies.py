import logging
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from typing import Any, cast

import asyncpg
import click
import sqlparse
from aiohttp import ClientSession
from cachetools import Cache, LRUCache
from fastapi import HTTPException
from injection import scoped, singleton
from sqlalchemy import URL, Connection, Executable, NullPool, event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.util import immutabledict
from starlette.status import HTTP_403_FORBIDDEN

from generic_rag.app.settings import ApplicationSettings
from generic_rag.channel import Channel
from generic_rag.db.auth import MsiTokenProvider, TokenProvider
from generic_rag.dial_client import DialClient
from generic_rag.scope import DialApplicationId, RequestApiKey, ScopeName
from generic_rag.services.channel_service import ChannelService
from generic_rag.types import FileStorage, ModelProvider

logger = logging.getLogger(__name__)


@singleton
async def token_provider_factory(
    settings: ApplicationSettings, exit_stack: AsyncExitStack
) -> TokenProvider | None:
    return await exit_stack.enter_async_context(MsiTokenProvider()) if settings.database.msi_enabled else None


@singleton
async def asyncpg_pool_factory(
    settings: ApplicationSettings, token_provider: TokenProvider | None, exit_stack: AsyncExitStack
) -> asyncpg.Pool:
    db_url = URL.create(
        drivername="postgresql",
        host=settings.database.host,
        port=settings.database.port,
        database=settings.database.dbname,
        username=settings.database.username,
        password=settings.database.password.get_secret_value()
        if settings.database.password and not settings.database.msi_enabled
        else None,
    )

    pool = await exit_stack.enter_async_context(
        asyncpg.create_pool(
            dsn=db_url.render_as_string(hide_password=False),
            password=(lambda: token_provider.token) if token_provider is not None else None,
            max_inactive_connection_lifetime=300,
        )
    )

    async with pool.acquire():
        logger.info(f"Connected to DB at: {db_url}")

    return pool


@singleton
async def async_engine(pool: asyncpg.Pool, exit_stack: AsyncExitStack) -> AsyncEngine:
    engine = create_async_engine(url="postgresql+asyncpg://", async_creator=pool.acquire, poolclass=NullPool)
    exit_stack.push_async_callback(engine.dispose)

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

    return engine


@singleton
class DialClientFactory:
    _cache: Cache | None = None

    def __init__(self, settings: ApplicationSettings, http_session: ClientSession):
        self._settings = settings
        self._http_session = http_session

        if settings.in_memory_cache.enabled:
            self._cache = LRUCache(
                maxsize=int(self._settings.in_memory_cache.capacity),
                getsizeof=len,
            )

    def create(self, request_api_key: RequestApiKey) -> DialClient:
        return DialClient(
            self._http_session,
            self._settings.dial_url.encoded_string(),
            request_api_key.get_secret_value(),
            in_memory_cache=self._cache,
        )


@singleton
def http_session_factory(exit_stack: AsyncExitStack) -> ClientSession:
    session = ClientSession()
    exit_stack.push_async_exit(session)
    return session


@scoped(ScopeName.channel)
def dial_client_factory(factory: DialClientFactory, request_api_key: RequestApiKey) -> DialClient:
    return factory.create(request_api_key)


@scoped(ScopeName.channel)
def model_provider_factory(dial_client: DialClient) -> ModelProvider:
    return cast(ModelProvider, dial_client)


@scoped(ScopeName.channel)
def file_storage_factory(dial_client: DialClient) -> FileStorage:
    return dial_client.get_file_storage()


@scoped(ScopeName.channel)
async def channel_factory(application_id: DialApplicationId, channel_service: ChannelService) -> Channel:
    if not application_id:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Forbidden",
        )
    return await channel_service.get_channel(application_id)
