from collections.abc import Sequence
from typing import Annotated, Any, Literal

from injection import scoped
from pydantic import BaseModel, ConfigDict, Field, create_model

from generic_rag.channel import Channel, RequestConfig
from generic_rag.scope import ScopeName
from generic_rag.types import AnyChunk, ImageChunk, RetrievedDocument, Retriever, TextChunk
from generic_rag.utils.answers import PlainAnswer


class RetrievalRequest(BaseModel):
    """Data retrieval request options."""

    query: Annotated[str, Field(description="The search query")]
    retriever: Annotated[dict[str, Any], Field(default_factory=dict)]

    @classmethod
    async def get_dynamic_model(cls):
        retriever_config_model = await Retriever.get_aggregated_config_model()

        # noinspection PyTypeChecker
        return create_model(
            cls.__name__,
            __base__=cls,
            __doc__=cls.__doc__,
            retriever=Annotated[
                retriever_config_model | None,
                Field(default=None, description="Retriever configuration overrides"),
            ],
        )


class SingleResult[T](BaseModel):
    result_type: T

    source_url: str = Field(description="url of the chunk source")
    source_display_name: str = Field(description="name of the chunk source that can be displayed to user")
    source_metadata: dict = Field(default_factory=dict, description="metadata associated with chunk source")


class TextChunkResult(TextChunk, SingleResult[Literal["chunk"]]):
    result_type: Literal["chunk"] = "chunk"


class ImageChunkResult(ImageChunk, SingleResult[Literal["chunk"]]):
    result_type: Literal["chunk"] = "chunk"
    model_config = ConfigDict(
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )


class CombinedResult(SingleResult[Literal["combined"]]):
    """Retrieval result that is combined of multiple chunks."""

    result_type: Literal["combined"] = "combined"
    chunks: Sequence[TextChunk | ImageChunk]


type SingleChunkResult = Annotated[
    TextChunkResult | ImageChunkResult,
    Field(discriminator="chunk_type", description="Retrieval result that consist of single chunk."),
]
type RetrievalResult = Annotated[
    CombinedResult | SingleChunkResult,
    Field(discriminator="result_type"),
]


@scoped(ScopeName.channel)
class RetrievalService:
    def __init__(self, channel: Channel):
        self._channel = channel

    @staticmethod
    async def get_request_schema() -> dict[str, Any]:
        retrieval_request_model = await RetrievalRequest.get_dynamic_model()
        return retrieval_request_model.model_json_schema()

    async def data_retrieval(self, request: RetrievalRequest) -> Sequence[RetrievalResult]:
        """Run retriever and return chunks which are relevant to a given query."""
        request_config_model = await RequestConfig.get_dynamic_model()
        request_config = request_config_model.create(
            defaults=self._channel.request_config, overrides={"retriever": request.retriever}
        )

        retriever = Retriever.create(request_config.retriever)

        return [self._convert(doc) for doc in await retriever.invoke(request.query, PlainAnswer())]

    def _convert(self, doc: RetrievedDocument) -> RetrievalResult:
        if len(doc.chunks) > 1:
            return CombinedResult(
                source_url=doc.source_url,
                source_display_name=doc.source_display_name,
                source_metadata=doc.source_metadata,
                chunks=[self._convert_chunk(chunk, doc) for chunk in doc.chunks],
            )

        return self._convert_chunk(doc.chunks[0], doc)

    @staticmethod
    def _convert_chunk(chunk: AnyChunk, doc: RetrievedDocument) -> TextChunkResult | ImageChunkResult:
        if isinstance(chunk, TextChunk):
            return TextChunkResult(
                source_url=doc.source_url,
                source_display_name=doc.source_display_name,
                source_metadata=doc.source_metadata,
                **chunk.model_dump(),
            )

        if isinstance(chunk, ImageChunk):
            return ImageChunkResult(
                source_url=doc.source_url,
                source_display_name=doc.source_display_name,
                source_metadata=doc.source_metadata,
                **chunk.model_dump(),
            )

        raise RuntimeError(f"unexpected chunk type: {type(chunk)}")
