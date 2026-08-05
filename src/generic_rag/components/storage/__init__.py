import logging
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager

from elastic_transport import TransportError
from elasticsearch import AsyncElasticsearch
from injection import aget_instance, singleton
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from generic_rag.app.settings import ApplicationSettings
from generic_rag.components.storage.elasticsearch import ElasticsearchIndexStorageBackend
from generic_rag.components.storage.pgvector import PgvectorIndexStorageBackend
from generic_rag.db.session import DbSessionMaker
from generic_rag.types import IndexStorageBackend

logger = logging.getLogger(__name__)


@singleton
async def pgvector_storage_backend_factory() -> PgvectorIndexStorageBackend:
    @asynccontextmanager
    async def create_session() -> AsyncGenerator[AsyncSession]:
        session = DbSessionMaker()

        async with session.begin():
            yield session

    return await PgvectorIndexStorageBackend.create(
        create_session  # todo: use separate db connection
    )


@singleton
async def elasticsearch_storage_backend_factory(
    settings: ApplicationSettings, exit_stack: AsyncExitStack
) -> ElasticsearchIndexStorageBackend | None:
    if not settings.elasticsearch:
        return None

    url = settings.elasticsearch.url
    es_client = await exit_stack.enter_async_context(
        AsyncElasticsearch(
            hosts=f"{url.scheme}://{url.host}:{url.port}/",
            basic_auth=(settings.elasticsearch.username, settings.elasticsearch.password.get_secret_value()),
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
