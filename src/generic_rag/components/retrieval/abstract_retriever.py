import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Collection, Sequence
from typing import Annotated, NotRequired, Self, TypedDict, cast

import tabulate
from annotated_types import MinLen
from injection import inject
from pydantic import BaseModel, Field, NonNegativeInt, TypeAdapter, create_model

from generic_rag.channel import Channel
from generic_rag.components.retrieval.document_selector import (
    AllDocumentsDocumentSelector,
    DocumentSelector,
)
from generic_rag.components.search_index import ChunkIndex, tracer
from generic_rag.services.chunk_service import ChunkService
from generic_rag.services.document_service import DocumentService
from generic_rag.types import AbstractAnswer, AnswerStage, AnyChunk, Document, RetrievedDocument, Retriever
from generic_rag.utils.answers import NoopStage

logger = logging.getLogger(__name__)


class FailedRetrieverError(Exception): ...


class AbstractRetrieverRequest(BaseModel, ABC):
    """Request-specific options of :class:`AbstractRetriever`."""

    top_k: dict[str, int | None]

    @classmethod
    async def get_dynamic_model(cls, channel: Channel) -> type["AbstractRetrieverRequest"]:
        # noinspection PyTypedDict
        top_k_fields = {
            idx.index_name: Annotated[
                NotRequired[NonNegativeInt],
                Field(description=f"Maximum number of results to be returned by `{idx.index_name}` index."),
            ]
            for idx in await channel.get_indexes()
        }
        # noinspection PyTypedDict
        top_k_model = TypedDict(AbstractRetrieverRequest.__name__ + "TopK", top_k_fields)
        top_k_default = {idx.index_name: idx.default_limit for idx in await channel.get_indexes()}

        return create_model(
            cls.__name__,
            __base__=cls,
            __doc__=cls.__doc__,
            top_k=Annotated[
                top_k_model,
                MinLen(1),
                Field(
                    top_k_default,
                    description=(
                        "Overrides for maximum number of results to be returned by given index. For the index "
                        "to be enabled for search corresponding non-zero value should be explicitly defined."
                    ),
                ),
            ],
        )

    @classmethod
    async def get_default_value(cls, channel: Channel) -> Self:
        model = await cls.get_dynamic_model(channel)
        return model.model_validate({})


class AbstractRetrieverConfig(BaseModel, ABC):
    """:class:`AbstractRetriever` configuration model."""

    document_selector: BaseModel

    @classmethod
    @inject
    async def get_dynamic_model(cls, channel: Channel | None = None) -> type["AbstractRetrieverConfig"]:
        document_selector_model = await DocumentSelector.get_aggregated_config_model()
        document_selector_default = TypeAdapter(document_selector_model).validate_python({
            "type": AllDocumentsDocumentSelector.get_qualifier()
        })

        base = cls if channel is None else (cls, await AbstractRetrieverRequest.get_dynamic_model(channel))

        # noinspection bad-return
        return create_model(
            cls.__name__,
            __doc__=cls.__doc__,
            __base__=base,
            document_selector=Annotated[
                document_selector_model,
                Field(
                    document_selector_default,
                    description="Allows to restrict search scope by selecting subset of documents.",
                ),
            ],
        )


class AbstractRetriever[ConfigT: AbstractRetrieverConfig = AbstractRetrieverConfig](Retriever[ConfigT], ABC):
    """Base class for retrievers implementation."""

    @inject
    def __init__(
        self,
        config: ConfigT,
        channel: Channel = NotImplemented,
        document_service: DocumentService = NotImplemented,
        chunk_service: ChunkService = NotImplemented,
    ):
        super().__init__(config)

        self._channel = channel
        self._document_service = document_service
        self._chunk_service = chunk_service

        self._lock = asyncio.Lock()
        self._sources: dict[int, Document] = {}
        """ cache of previously loaded sources """

    async def invoke(self, query: str, answer: AbstractAnswer) -> list[RetrievedDocument]:
        """
        Retrieve pieces of information that are relevant to given query.

        :param query: the user query used for search
        :param answer: the current answer (to report retrieval stages execution)
        """
        combined_search_name = f"Combined search ({self.get_qualifier()})"
        combined_search_stage = answer.create_stage(combined_search_name)

        documents = await DocumentSelector.create(self.config.document_selector).get_document_subset(answer)
        if documents is not None and len(documents) < 1:
            return []

        async def _combined_search() -> Sequence[RetrievedDocument]:
            request_config = (
                cast(AbstractRetrieverRequest, self.config)
                if isinstance(self.config, AbstractRetrieverRequest)
                else await AbstractRetrieverRequest.get_default_value(self._channel)
            )
            tasks = [
                asyncio.create_task(
                    _run_retrieval_stage(
                        answer.create_stage(index.display_name),
                        index.display_name,
                        self._index_search(query, index, top_k, documents),
                    )
                )
                for index in await self._channel.get_indexes()
                if (top_k := (request_config.top_k.get(index.index_name) or 0))
            ]
            if tasks:
                doc_lists = await asyncio.gather(*tasks)
                return await self._get_combined_results(doc_lists)
            return []

        return list(
            await _run_retrieval_stage(combined_search_stage, combined_search_name, _combined_search())
        )

    async def _index_search(
        self, query: str, index: ChunkIndex, top_k: int, documents: list[int] | None = None
    ) -> Sequence[RetrievedDocument]:
        """
        Run search in given index.

        :param query: the user query
        :param index: the index to search in
        :param top_k: maximum number of results to return
        :param documents: limit the search to the given subset of documents (if not `None`)
        """
        chunk_refs = await index.search(query, limit=top_k, documents=documents)

        return [
            RetrievedDocument(
                chunks=[chunk],
                source_id=source.id,
                source_url=source.url,
                source_page_number=chunk.page_number,
                source_display_name=source.display_name,
                source_metadata=source.metadata,
            )
            for chunk, source in await self._load_sources(
                await self._chunk_service.get_chunks_by_references(chunk_refs)
            )
        ]

    async def _load_sources(self, chunks: Collection[AnyChunk]) -> Sequence[tuple[AnyChunk, Document]]:
        """
        Load sources for given chunks.

        :param chunks: list of found chunks
        :return: list of pairs (chunk,document)
        """
        async with self._lock:
            if missing_sources := [
                chunk.document_id for chunk in chunks if chunk.document_id not in self._sources
            ]:
                for document in await self._document_service.get_documents_by_id(missing_sources):
                    self._sources[document.id] = document

        return [(chunk, self._sources[chunk.document_id]) for chunk in chunks]

    @abstractmethod
    async def _get_combined_results(
        self, doc_lists: list[Sequence[RetrievedDocument]]
    ) -> Sequence[RetrievedDocument]:
        """Combine given results into single list."""


@tracer.start_as_current_span("retriever-stage")
async def _run_retrieval_stage(
    stage: AnswerStage, stage_name: str, callback: Awaitable[Sequence[RetrievedDocument]]
) -> Sequence[RetrievedDocument]:
    try:
        with stage:
            logger.info(f"Running stage '{stage_name}'")
            try:
                result = await callback
            except Exception as e:
                raise FailedRetrieverError(str(e)) from e

            stage.append_content(f"Found {len(result)} reference(s)\n\n")

            if not isinstance(stage, NoopStage):
                if chunk_summary := _get_chunks_summary(result):
                    stage.append_content(
                        tabulate.tabulate(chunk_summary, headers="keys", tablefmt="html") + "\n\n"
                    )

                for i, document in enumerate(result, start=1):
                    await stage.add_reference(i, document)

        return result

    except FailedRetrieverError as e:
        logger.warning(f"Internal error in '{stage_name}': {str(e)}", exc_info=e)
        return []


def _get_chunks_summary(retrieved_docs: Sequence[RetrievedDocument]):
    result = []

    for i, document in enumerate(retrieved_docs, start=1):
        result.extend(
            {
                "#": f"[{i}]",
                "source_name": document.source_display_name,
                "page_number": chunk.page_number,
            }
            | (document.model_extra or {})
            | chunk.get_identity().model_dump()
            for chunk in document.chunks
        )

    return result
