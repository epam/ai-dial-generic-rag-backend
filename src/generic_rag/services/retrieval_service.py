from collections.abc import Sequence
from typing import Annotated, Any, Literal

from injection import scoped
from pydantic import BaseModel, ConfigDict, Field, create_model

from generic_rag.channel import Channel, RequestConfig
from generic_rag.scope import ScopeName
from generic_rag.services.chunk_sources_manager import ChunkSource
from generic_rag.types import AnyChunk, ImageChunk, Retriever, TextChunk


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


class TextChunkResult(ChunkSource, TextChunk):
    result_type: Literal["chunk"] = "chunk"


class ImageChunkResult(ChunkSource, ImageChunk):
    result_type: Literal["chunk"] = "chunk"

    model_config = ConfigDict(
        val_json_bytes="base64",
        ser_json_bytes="base64",
    )


type SingleChunkResult = Annotated[
    TextChunkResult | ImageChunkResult,
    Field(discriminator="chunk_type", description="Retrieval result that consist of single chunk."),
]


class CombinedResult(ChunkSource):
    """Retrieval result that is combined of multiple chunks."""

    result_type: Literal["combined"] = "combined"
    chunks: Sequence[AnyChunk]

    model_config = ConfigDict(
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )


type RetrievalResult = Annotated[SingleChunkResult | CombinedResult, Field(discriminator="result_type")]


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

        return [
            self._convert(chunks)
            for doc in await retriever.invoke(request.query)
            if (chunks := doc.metadata.get("chunks", []))
        ]

    @staticmethod
    def _convert(chunks: list[AnyChunk]) -> RetrievalResult:
        if not chunks:
            raise ValueError("chunks is empty")

        chunk_source = ChunkSource.from_chunk(chunks[0])

        if len(chunks) > 1:
            return CombinedResult.model_validate(
                dict(
                    chunks=[
                        chunk.model_dump(exclude=set(chunk.model_extra.keys())) if chunk.model_extra else None
                        for chunk in chunks
                    ],
                    **chunk_source.model_dump(),
                )
            )

        if isinstance(chunks[0], TextChunk):
            return TextChunkResult.model_validate(chunks[0].model_dump())

        return ImageChunkResult.model_validate(chunks[0].model_dump())
