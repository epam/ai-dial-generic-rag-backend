import logging
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import cast

from aiohttp import ClientSession
from cachetools import Cache, LRUCache
from elastic_transport import TransportError
from elasticsearch import AsyncElasticsearch
from fastapi import HTTPException
from injection import aget_instance, scoped, singleton
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.status import HTTP_403_FORBIDDEN
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from generic_rag.app.settings import ApplicationSettings
from generic_rag.channel import Channel
from generic_rag.components.storage.elasticsearch import ElasticsearchIndexStorageBackend
from generic_rag.components.storage.pgvector import PgvectorIndexStorageBackend
from generic_rag.db.session import DbSessionMaker
from generic_rag.dial_client import DialClient
from generic_rag.scope import DialApplicationId, RequestApiKey, ScopeName
from generic_rag.services.channel_service import ChannelService
from generic_rag.types import FileStorage, IndexStorageBackend, ModelProvider

logger = logging.getLogger(__name__)


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


@singleton
async def pgvector_storage_backend_factory() -> PgvectorIndexStorageBackend:
    @asynccontextmanager
    async def create_session() -> AsyncGenerator[AsyncSession]:
        session = DbSessionMaker()

        async with session.begin():
            yield session

    return await PgvectorIndexStorageBackend.create(create_session)


@singleton
async def elasticsearch_storage_backend_factory(settings: ApplicationSettings, exit_stack: AsyncExitStack) -> (
    ElasticsearchIndexStorageBackend | None
):
    if not settings.elasticsearch:
        return None

    url = settings.elasticsearch.url
    es_client = await exit_stack.enter_async_context(
        AsyncElasticsearch(
            hosts=f"{url.scheme}://{url.host}:{url.port}/",
            basic_auth=(
                settings.elasticsearch.username,
                settings.elasticsearch.password.get_secret_value()
            ),
        )
    )

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(), retry=retry_if_exception_type(TransportError))
    async def _check_connection():
        await es_client.info()

    try:
        logger.info(f"Connecting to Elasticsearch at: {es_client}")
        await _check_connection()
    except TransportError as e:
        logger.warning(f"Could not connect to Elasticsearch: {str(e)}")
        await es_client.close()
        raise
    else:
        logger.info(f"Connected to Elasticsearch at: {es_client}")

        return ElasticsearchIndexStorageBackend(
            es_client,
            settings.elasticsearch.index_prefix,
        )


@singleton
async def index_storage_backends_factory() -> dict[str, IndexStorageBackend]:
    return {
        impl.get_qualifier(): instance
        for impl in IndexStorageBackend.get_implementations()
        if (instance := await aget_instance(impl, None)) is not None
    }


@scoped(ScopeName.channel)
async def channel_factory(application_id: DialApplicationId, channel_service: ChannelService) -> Channel:
    if not application_id:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Forbidden",
        )
    return await channel_service.get_channel(
        application_id
    )
