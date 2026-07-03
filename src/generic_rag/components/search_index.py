import logging
import operator
from abc import ABC
from collections.abc import Collection, Iterable
from functools import reduce
from typing import Annotated, Literal

from injection import afind_instance, inject
from pydantic import BaseModel, Field, TypeAdapter, create_model

from generic_rag.types import (
    DEFAULT_BACKEND,
    DEFAULT_RESULTS_LIMIT,
    AnyChunk,
    ChunkRef,
    ConfigurableComponent,
    IndexedEntityMeta,
    Indexer,
    IndexStorage,
    IndexStorageBackend,
    TextType,
    VectorType,
)
from generic_rag.utils.profile import log_execution_time

logger = logging.getLogger(__name__)


class StorageBackendDiscriminator(BaseModel):
    backend: str


class IndexConfig(BaseModel, ABC):
    """:class:`Index` configuration model."""

    display_name: str = Field(..., description="Human-friendly name of the index.")
    indexer: BaseModel
    storage: StorageBackendDiscriminator
    default_limit: int = Field(
        DEFAULT_RESULTS_LIMIT,
        description="Default value for maximum number of results to be returned by search within this index.",
    )

    @classmethod
    @inject
    async def get_dynamic_model[T: IndexConfig](
        cls: type[T],
        index_backends: dict[str, IndexStorageBackend] = NotImplemented,
    ) -> type[T]:
        indexer_model = await Indexer.get_aggregated_config_model()

        # noinspection PyTypeHints
        backend_variants = [
            create_model(
                backend.storage_options_model.__name__,
                __base__=(
                    backend.storage_options_model,
                    create_model(
                        StorageBackendDiscriminator.__name__,
                        __base__=StorageBackendDiscriminator,
                        backend=Literal[key],
                    ),
                ),
                __doc__=backend.storage_options_model.__doc__,
            )
            for key, backend in index_backends.items()
        ]

        storage_model = reduce(operator.or_, backend_variants)
        storage_model_default = TypeAdapter(storage_model).validate_python({
            "backend": DEFAULT_BACKEND,
        })

        # noinspection PyTypeChecker
        return create_model(
            cls.__name__,
            __base__=cls,
            __doc__=cls.__doc__,
            indexer=indexer_model,
            storage=Annotated[
                storage_model,
                Field(
                    default=storage_model_default,
                    discriminator="backend",
                    description="Configuration of index storage for given index.",
                ),
            ],
        )


class Index[IndexT: TextType | VectorType, ConfigT: IndexConfig = IndexConfig](
    ConfigurableComponent[ConfigT]
):
    """Component that allows to perform relevancy search of previously indexed data."""

    _storage: IndexStorage[IndexT]

    def __init__(self, config: ConfigT, index_name: str, indexer: Indexer[IndexT]):
        super().__init__(config)
        self._index_name = index_name
        self._display_name = config.display_name
        self._default_limit = config.default_limit
        self._indexer = indexer

    @classmethod
    async def create_async[T: Index](cls: type[T], config: ConfigT, **kwargs) -> T:
        """Create instance of this component for given configuration."""
        channel_key = kwargs.get("channel_key")
        index_name = kwargs.get("index_name")

        assert isinstance(channel_key, str)
        assert isinstance(index_name, str)

        backends = await afind_instance(dict[str, IndexStorageBackend])
        indexer = Indexer.create(config.indexer)

        implementation = cls.create(config, index_name=index_name, indexer=indexer)

        if (backend := backends.get(config.storage.backend)) and (
            storage := await backend.get_storage(
                channel_key,
                index_name,
                indexer,
                config.storage,
            )
        ):
            implementation._storage = storage  # noqa: SLF001
            return implementation

        raise RuntimeError("Unable to create index")

    @property
    def index_name(self) -> str:
        """Name of the index."""
        return self._index_name

    @property
    def display_name(self):
        """Human-friendly name of the index."""
        return self._display_name

    @property
    def default_limit(self):
        """Default value for maximum number of results to be returned by search within this index."""
        return self._default_limit

    @property
    def storage(self) -> IndexStorage[IndexT]:
        """Storage of this index data"""
        return self._storage


class ChunkIndex[IndexT: TextType | VectorType, IndexerConfigT: BaseModel](Index[IndexT]):
    """Index allowing relevance search on top of document chunks."""

    @log_execution_time(logger)
    async def search(
        self, query: str, limit: int, documents: Collection[int] | None = None
    ) -> Collection[ChunkRef]:
        """
        Search records within the index.

        :param query: query used for search
        :param limit: maximum number of records to return
        :param documents: allows to define the subset of documents to use
        """
        indexed_query = await self._indexer.index_query(query)
        return [
            ChunkRef.model_validate(meta.model_dump())
            for meta in await self._storage.relevance_search(indexed_query, limit, documents)
        ]

    async def update(self, chunks: Iterable[AnyChunk]):
        """
        Update index with given chunks.

        :param chunks: chunks to update the index with
        """
        data = [
            (chunk, IndexedEntityMeta.model_validate(chunk.get_identity().model_dump())) for chunk in chunks
        ]

        records = await self._indexer.index_data(data)
        unique_documents = {record.metadata.document_id for record in records}

        await self._storage.remove(*unique_documents)
        await self._storage.add(records)
