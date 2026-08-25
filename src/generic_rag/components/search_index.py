import logging
import operator
from abc import ABC
from collections.abc import Collection, Iterable
from functools import reduce
from typing import Annotated, Literal

import jsonpath_ng
from injection import afind_instance, inject
from opentelemetry.trace import get_tracer
from pydantic import BaseModel, Field, TypeAdapter, create_model

from generic_rag.types import (
    DEFAULT_BACKEND,
    DEFAULT_RESULTS_LIMIT,
    AnyChunk,
    ChunkRef,
    ConfigurableComponent,
    Document,
    IndexedEntityMeta,
    Indexer,
    IndexStorage,
    IndexStorageBackend,
    TextType,
    VectorType,
)
from generic_rag.utils.profile import log_execution_time

tracer = get_tracer(__name__)
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
    def storage(self) -> IndexStorage[IndexT]:
        """Storage of this index data"""
        return self._storage


class ChunkIndex[IndexT: TextType | VectorType](Index[IndexT]):
    """Enables relevance search and retrieval of documents content pieces (chunks)."""

    @log_execution_time(logger)
    async def search(
        self, query: str, limit: int | None = None, documents: Collection[int] | None = None
    ) -> Collection[ChunkRef]:
        """
        Search for the data within the index.

        :param query: query used for search
        :param limit: maximum number of records to return
        :param documents: allows to define the subset of documents to use
        """
        return [
            ChunkRef.model_validate(meta.model_dump())
            for meta in await self._storage.relevance_search(
                await self._indexer.index_query(query), limit or self.config.default_limit, documents
            )
        ]

    @tracer.start_as_current_span("index-update")
    @log_execution_time(logger)
    async def add(self, chunks: Iterable[AnyChunk]):
        """
        Add given chunks into index.

        :param chunks: chunks to update the index with
        """
        if data := [
            (
                chunk,
                IndexedEntityMeta.model_validate(
                    chunk.get_identity().model_dump(),
                ),
            )
            for chunk in chunks
        ]:
            await self._storage.add(
                await self._indexer.index_data(data),
            )


class DocumentIndexConfig(IndexConfig):
    fields: list[
        Annotated[
            str, Field(pattern=r"^\$(?:\.[a-zA-Z_][a-zA-Z0-9_*]*|\?|\[(?:[0-9*]+|'[^']+'|\"[^\"]+\")\])*$")
        ]
    ] = Field(
        description=(
            "List of fields to include into the index, defined using JSON-path syntax. "
            "The following fields can be used:\n"
            "* `$.display_name`\n"
            "* `$.content` (for text-based documents only)\n"
            "* any string field of document's metadata (`$.metadata.<field_name>`)"
        ),
    )
    include_in_hybrid: bool = Field(
        True,
        description="Indicates that this index should be included by default in hybrid search.",
    )


class DocumentIndex[IndexT: TextType | VectorType](Index[IndexT, DocumentIndexConfig]):
    """Enables relevance search of documents based on their fields."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fields = [jsonpath_ng.parse(field_expr) for field_expr in self.config.fields]

    @tracer.start_as_current_span("index-search")
    @log_execution_time(logger)
    async def search(
        self, query: str, limit: int | None = None, documents: Collection[int] | None = None
    ) -> Collection[int]:
        """
        Search for the data within the index.

        :param query: query used for search
        :param limit: maximum number of records to return
        :param documents: allows to define the subset of documents to use
        """
        return [
            meta.document_id
            for meta in await self._storage.relevance_search(
                await self._indexer.index_query(query), limit or self.config.default_limit, documents
            )
        ]

    @tracer.start_as_current_span("index-update")
    @log_execution_time(logger)
    async def add(self, documents: Iterable[Document]):
        """Add given documents to the index."""
        if data := [
            (
                item,
                IndexedEntityMeta(
                    document_id=doc.id,
                ),
            )
            for doc in documents
            if (item := await self._extract_data(doc))
        ]:
            await self._storage.add(
                await self._indexer.index_data(data),
            )

    async def _extract_data(self, doc: Document) -> str:
        document_view = {"display_name": doc.display_name, "metadata": doc.metadata}
        if doc.mime_type.lower().startswith("text/"):
            document_view["content"] = (await doc.get_content()).decode()
        return "\n".join([
            match.value
            for field in self._fields
            for match in field.find(document_view)
            if isinstance(match.value, str) and len(match.value)
        ])
