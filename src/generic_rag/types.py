import asyncio
import datetime
import enum
import operator
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Collection, Generator, Iterable
from contextlib import AbstractContextManager
from enum import StrEnum
from functools import cache, reduce
from types import UnionType
from typing import (
    Annotated,
    Any,
    BinaryIO,
    ClassVar,
    Literal,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import humps
from datauri import DataURI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import BaseModel, BeforeValidator, ByteSize, ConfigDict, Field, create_model
from pydantic_core import PydanticUndefined

from generic_rag.utils.generics import resolve_generic_arg

DEFAULT_BACKEND = "pgvector"  # todo: get rid of hardcode
DEFAULT_RESULTS_LIMIT = 7
DEFAULT_LLM_DEPLOYMENT = "gpt-4.1-2025-04-14"


@enum.unique
class ChunkType(StrEnum):
    text = enum.auto()
    image = enum.auto()


class ChunkRef[T: ChunkType](BaseModel):
    chunk_type: T = Field(..., description="type of this chunk")
    document_id: int = Field(..., description="id of document where this chunks originates")
    chunk_id: int = Field(..., description="id of the chunk within the document")

    model_config = ConfigDict(
        frozen=True,
    )

    def get_identity(self) -> "ChunkRef":
        if type(self) is ChunkRef:
            return self
        return ChunkRef(
            chunk_type=self.chunk_type,
            document_id=self.document_id,
            chunk_id=self.chunk_id,
        )


class ChunkMetadata(BaseModel):
    """Additional information related to a chunk."""

    page_number: int = Field(description="number of page where this chunk was extracted")


class TextChunk(ChunkMetadata, ChunkRef):
    """A piece of text information."""

    chunk_type: Literal[ChunkType.text] = ChunkType.text
    text: str


@enum.unique
class ImageType(StrEnum):
    """Type of image stored by image chunk."""

    page = enum.auto()
    table = enum.auto()
    diagram = enum.auto()


class ImageChunk(ChunkMetadata, ChunkRef):
    """A piece of graphical information."""

    chunk_type: Literal[ChunkType.image] = ChunkType.image
    image_type: ImageType = Field(..., description="type of this chunk's image")
    mime_type: str = Field(..., description="mime type of this chunk's content")
    content: bytes = Field(..., description="the content of this chunk", repr=False)

    model_config = ConfigDict(
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )

    def get_data_uri(self) -> DataURI:
        return DataURI.make(
            self.mime_type,
            charset=None,
            base64=True,
            data=self.content,
        )


type AnyChunk = TextChunk | ImageChunk


@enum.unique
class DocumentStatus(StrEnum):
    created = enum.auto()
    processing = enum.auto()
    processed = enum.auto()
    indexing = enum.auto()
    ready = enum.auto()
    error = enum.auto()


class Document(BaseModel, ABC):
    """
    A document stored in the system
    """

    id: int = Field(..., description="unique id of this document within the channel")
    url: str = Field(..., description="url of the original document")
    display_name: str = Field(..., description="user-facing name of the document")
    mime_type: str = Field(..., description="mime type of the original document")
    size: int = Field(..., description="size of the original document (in bytes)")
    metadata: dict = Field(
        default_factory=dict,
        description="optional metadata associated with this document (should match the schema associated with the channel)",
    )
    status: DocumentStatus = Field(..., description="the status of this document processing")

    async def get_content(self) -> bytes | None:
        """Returns the document's content."""
        if (stream := await self.get_content_stream()) is not None:
            return b"".join([chunk async for chunk in stream])
        return None

    @abstractmethod
    async def get_content_stream(self) -> AsyncIterable[bytes] | None:
        """Returns async iterable on chunks of the document's content."""


class Component(ABC):  # noqa: B024
    """Component is a main building block for RAG pipeline."""

    __qualifier: ClassVar[str | None] = None

    @classmethod
    def get_qualifier(cls) -> str:
        """Return a qualifier - a special key that can be used to identify this component."""
        if not cls.__qualifier:
            name = cls.__name__
            for candidate in cls.mro():
                if candidate is cls or candidate is Component or not issubclass(candidate, Component):
                    continue
                name = name.removesuffix(candidate.__name__)
            cls.__qualifier = humps.depascalize(name)
        return cls.__qualifier

    @classmethod
    def get_implementations[T](cls: type[T]) -> Generator[type[T]]:
        """Recursively iterate over implementations of this component."""
        for child in cls.__subclasses__():
            assert issubclass(child, Component)
            if issubclass(child, ABC):
                if not child.__abstractmethods__:
                    yield child
            else:
                yield child

            yield from child.get_implementations()


@runtime_checkable
class DynamicModel(Protocol):
    """Protocol used by :class:`ConfigurableComponent` to recognize dynamic configuration model."""

    @classmethod
    async def get_dynamic_model[T: BaseModel](cls: type[T]) -> type[T]:
        """Create dynamic (runtime) model for given class."""


class ConfigurableComponent[ConfigT: BaseModel = BaseModel](Component, ABC):
    """
    A component with configuration.

    The **configuration** is a pydantic model defined as generic argument. It can be static or dynamic.

    **Static** configuration described as the model class itself and does not require any additional actions:

    >>> class StaticConfig(BaseModel):
    >>>     property: str = Field(...)
    >>>
    >>> class MyComponent(ConfigurableComponent[StaticConfig]):
    >>>     ...
    >>>
    >>> # create component
    >>> my_component = ConfigurableComponent.create(
    >>>     StaticConfig.model_validate({"property": "some value"})
    >>> )

    To define **dynamic** configuration the model class should be compatible with :class:`DynamicModel`
    protocol, whose :meth:`DynamicModel.get_dynamic_model` method should return the actual model class.
    This approach especially useful when it's necessarry the model definition requires information which
    is not available during import time, or depend on some conditions.

    The example of dynamic configuration:

    >>> class DynamicConfig(BaseModel, ABC):
    >>>     dynamic_property: str
    >>>
    >>>     @classmethod
    >>>     async def get_dynamic_model(cls) -> type["DynamicConfig"]:
    >>>         return create_model(
    >>>             cls.__name__ + "Dynamic",
    >>>             __base__=cls,
    >>>             __doc__=cls.__doc__,
    >>>             dynamic_property=(Literal["first", "second", "third"], Field(...)),
    >>>         )
    >>>
    >>> class MyComponent(ConfigurableComponent[DynamicConfig]):
    >>>     ...
    >>>
    >>> # create component
    >>> dynamic_config_model = await DynamicConfig.get_dynamic_model()
    >>> my_component = ConfigurableComponent.create(
    >>>     dynamic_config_model.model_validate({"dynamic_property": "second"})
    >>> )
    """

    def __init__(self, config: ConfigT):
        self.__config = config

    @property
    def config(self) -> ConfigT:
        """Configuration of this component."""
        return self.__config

    @classmethod
    def create[T: ConfigurableComponent](cls: type[T], config: ConfigT, **kwargs) -> T:
        """Create instance of this component for given configuration."""
        for implementation in cls.get_implementations():
            if isinstance(config, implementation.get_config_model()):
                return implementation(config, **kwargs)
        raise RuntimeError(f"unable to find {cls.__name__} implementation for provided config: {config!r}")

    @classmethod
    @cache
    def get_config_model(cls) -> type[ConfigT]:
        """Return configuration model type defined for this component."""
        if result := resolve_generic_arg(cls, ConfigurableComponent, 0):
            if isinstance(result, TypeVar) and issubclass(result.__default__, BaseModel):
                result = result.__default__
            if result is BaseModel:
                # each component should have its own unique model class for config; otherwise, the logic
                # of selecting implementation based on the type of actual config instance will be broken
                return create_model(cls.__name__ + "Default", __base__=BaseModel)
            if issubclass(result, BaseModel):
                return result
        raise RuntimeError(f"unable to determine configuration model type for {cls}")

    @classmethod
    async def get_aggregated_config_model(
        cls,
        discriminator: str = "type",
        default_impl: type["ConfigurableComponent[ConfigT]"] | None = None,
    ) -> UnionType | None:
        """
        Get aggregated config model including models for all implementations of this component.

        The resulting model is created as pydantic's discriminated union with string discriminator,
        values of discriminator tags are determined with invocation of :meth:`Component.get_qualifier` method.

        :param discriminator: the name of discriminator field that will be resulting model's members
        :param default_impl: if set, given implementation will be used by default if discriminator omitted
        """
        assert default_impl is None or issubclass(default_impl, cls)

        async def _create_model_variant(impl: type[ConfigurableComponent[ConfigT]]) -> type[ConfigT]:
            # noinspection PyTypeHints
            fields = {
                discriminator: (
                    Literal[impl.get_qualifier()],
                    Field(impl.get_qualifier() if impl is default_impl else PydanticUndefined),
                )
            }
            config_model = impl.get_config_model()

            assert issubclass(config_model, BaseModel)
            assert config_model is not BaseModel

            if issubclass(config_model, DynamicModel):
                config_model = await config_model.get_dynamic_model()

            model_name = impl.__name__ + "Config"
            return create_model(
                model_name,
                __base__=(
                    config_model,
                    create_model(model_name + discriminator.capitalize(), **fields),
                ),
                __doc__=impl.__doc__ or config_model.__doc__,
                __config__=ConfigDict(
                    frozen=True,
                ),
            )

        def set_default_discriminator(value: Any) -> Any:
            if default_impl is not None and isinstance(value, dict) and discriminator not in value:
                value[discriminator] = default_impl.get_qualifier()
            return value

        tasks = [_create_model_variant(impl) for impl in cls.get_implementations()]

        if variants := tuple(await asyncio.gather(*tasks)):
            return Annotated[
                reduce(operator.or_, variants),
                Field(discriminator=discriminator, description=cls.__doc__),
                BeforeValidator(set_default_discriminator),
            ]

        return None


class DocumentParser[ConfigT: BaseModel = BaseModel](ConfigurableComponent[ConfigT], ABC):
    """Component used to extract chunks from document."""

    @abstractmethod
    async def extract_chunks(self, document: Document) -> AsyncIterable[AnyChunk]:
        """Extract chunks from given document"""


type TextType = str
""" Type for index values encoded as text (to be persisted in search engines like Elasticsearch). """

type VectorType = list[float]
""" Type for index values encoded as vector (for example, embeddings). """


class IndexedEntityMeta(BaseModel):
    """Metadata of indexed entity."""

    document_id: int = Field(..., description="id of the document")

    model_config = ConfigDict(
        extra="allow",
        frozen=True,
    )


class IndexRecord[IndexT: TextType | VectorType](BaseModel):
    """
    Single record stored in an index along with associated metadata.
    """

    index: Annotated[IndexT, Field(description="the record's content")]
    metadata: Annotated[
        IndexedEntityMeta, Field(description="the metadata of the indexed entity the record belongs to")
    ]


class Indexer[IndexT: TextType | VectorType, ConfigT: BaseModel = BaseModel](
    ConfigurableComponent[ConfigT], ABC
):
    """Component that performs actual indexing of data and user queries."""

    @abstractmethod
    async def index_query(self, query: str) -> IndexT:
        """Index given query in a way that allows to perform lookup for matching records."""

    @abstractmethod
    async def index_data(
        self, data: Iterable[tuple[AnyChunk | str, IndexedEntityMeta]]
    ) -> Collection[IndexRecord[IndexT]]:
        """Index given data for further storage."""


class IndexStorage[IndexT: TextType | VectorType](ABC):
    """Storage for single index."""

    @abstractmethod
    async def relevance_search(
        self,
        query: IndexT,
        limit: int,
        documents: Collection[int] | None = None,
    ) -> Collection[IndexedEntityMeta]:
        """
        Search for entities that are relevant to the given query.

        :param query: the indexed query used for search
        :param limit: maximum number of results to return
        :param documents: if set, the scope of search will be limited only to given documents
        :return: collection of the most relevant entities metadata (sorted by relevancy)
        """

    @abstractmethod
    async def add(self, records: Iterable[IndexRecord[IndexT]]):
        """
        Add given index records to the storage.

        :param records: record to update index with
        """

    @abstractmethod
    async def remove(self, *documents: int):
        """Remove index records for documents with given IDs."""

    @abstractmethod
    def export(self, *documents: int) -> AsyncIterable[IndexRecord[IndexT]]:
        """Export index records for documents with given IDs."""


class IndexStorageBackend[StorageOptionsT: BaseModel = BaseModel](Component, ABC):
    @abstractmethod
    async def get_storage[IndexT: TextType | VectorType](
        self,
        channel_key: str,
        index_name: str,
        indexer: Indexer[IndexT, BaseModel],
        options: StorageOptionsT | None = None,
    ) -> IndexStorage[IndexT]:
        """
        Return index storage for given channel, index name and indexer.

        :param channel_key: the key of the channel
        :param index_name: name of the index (should be unique within the channel)
        :param indexer: the indexer object
        :param options: additional options for the storage (depend on implementation)
        """

    @property
    @abstractmethod
    def storage_options_model(self) -> type[StorageOptionsT]:
        """ " The model of storage options for this backend."""


class IndexerCompatibilityError(RuntimeError):
    def __init__[IndexT: TextType | VectorType, IndexerConfigT: BaseModel](
        self,
        backend: IndexStorageBackend[Any],
        indexer: Indexer[IndexT, IndexerConfigT],
    ):
        super().__init__(
            f"backend '{backend.get_qualifier()}' cannot be used along with indexer '{indexer.get_qualifier()}'"
        )


class RetrievedDocument(BaseModel):
    """Single result of the retrieval."""

    chunks: list[AnyChunk] = Field(
        min_length=1, description="list of relevant chunks included in this result"
    )

    source_id: int = Field(description="ID of the source document")
    source_url: str = Field(description="URL of the source document")
    source_page_number: int = Field(description="number of page of the source document")
    source_display_name: str = Field(
        description="name of the source document that can be displayed to the user"
    )
    source_metadata: dict = Field(
        default_factory=dict, description="metadata associated with the source document"
    )

    model_config = ConfigDict(extra="allow")


class AnswerStage(AbstractContextManager, ABC):
    """Represents intermediate stage of RAG answer."""

    @abstractmethod
    def append_content(self, content: str): ...

    @abstractmethod
    async def add_citation(self, citation_index: int, doc: RetrievedDocument): ...

    @abstractmethod
    async def add_reference(self, citation_index: int, doc: RetrievedDocument): ...


class Answer(AnswerStage, ABC):
    """Represents the final RAG answer."""

    @abstractmethod
    def create_stage(self, name: str, *, debug: bool = False, timed: bool = True) -> AnswerStage: ...


class Retriever[ConfigT: BaseModel = BaseModel](ConfigurableComponent[ConfigT], ABC):
    """Top-level component responsible for retrieval of relevant chunks for further use."""

    @abstractmethod
    async def invoke(self, query: str, answer: Answer) -> list[RetrievedDocument]:
        """
        Retrieve pieces of information that are relevant to given query.

        :param query: the user query used for search
        :param answer: the current answer (to report retrieval stages execution)
        """


class AnswerGenerator[ConfigT: BaseModel = BaseModel](ConfigurableComponent[ConfigT], ABC):
    """Component that generates answer to a user query."""

    @abstractmethod
    async def invoke(self, query: str, retriever: Retriever, answer: Answer):
        """
        Generate answer to given user's query.

        :param query: the user query to answer
        :param retriever: the :class:`Retriever` used to find relevant chunk information
        :param answer: the current answer
        """


class LlmConfig(BaseModel):
    """Configuration for the LLM."""

    deployment_name: str = Field(
        default=DEFAULT_LLM_DEPLOYMENT,
        description="Name of LLM deployment to use.",
    )
    temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="Controls the randomness and creativity of a LLM's output."
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries to make when performing requests to the LLM.",
    )


class ModelProvider(ABC):
    @abstractmethod
    def get_embeddings_model(self, deployment: str, max_retries=3) -> AzureOpenAIEmbeddings:
        """
        Return embedding model with given deployment name.

        :param deployment: a name of model deployment
        :param max_retries: maximum number of retries to make when performing requests to the model
        """

    @abstractmethod
    def get_llm(self, config: LlmConfig) -> AzureChatOpenAI:
        """
        Return LLM for given configuration.

        :param config: the LLM configuration
        """


class FileMetadata(BaseModel):
    name: str = Field(..., description="File name")
    content_type: str = Field(..., description="File MIME type")
    content_length: ByteSize = Field(..., description="File size (in bytes)")
    created_at: datetime.datetime = Field(..., description="A timestamp of when the file was created")
    updated_at: datetime.datetime = Field(..., description="A timestamp of when the file was last updated")
    url: str = Field(..., description="URL of the file within the storage")
    etag: str = Field(..., description="Special tag used to files versioning")


class FileStorage(ABC):
    @abstractmethod
    async def get_bucket(self) -> str:
        """Name of a bucket for storing files."""

    @abstractmethod
    async def put_file(
        self, bucket: str, filepath: str, content_type: str, content: bytes | BinaryIO
    ) -> FileMetadata:
        """Put given content into a bucket and return the metadata of created file."""

    @abstractmethod
    async def get_file_metadata(self, url: str) -> FileMetadata | None:
        """Download metadata of file with given url, or return None if the file doesn't exist."""

    @abstractmethod
    async def download_file(self, url: str) -> AsyncIterable[bytes] | None:
        """Download content of file with given url, or return None if the file doesn't exist."""

    @abstractmethod
    async def copy_file_to_user(self, source_url: str, destination_name: str) -> str:
        """Copy file with given url to a user bucket, and return the url of copied file."""

    @abstractmethod
    async def delete_file(self, url: str):
        """Delete file with given url."""
