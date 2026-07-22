import enum
from collections.abc import AsyncGenerator
from enum import StrEnum
from typing import Self, cast

from injection import inject
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document as LangchainDocument
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from generic_rag.channel import Channel
from generic_rag.components.retrieval.abstract_retriever import AbstractRetriever, AbstractRetrieverConfig
from generic_rag.components.retrieval.index_retriever import IndexRetriever
from generic_rag.components.retrieval.retrieval_stage import RetrievalStage
from generic_rag.components.search_index import ChunkIndex
from generic_rag.services.chunk_service import ChunkService
from generic_rag.types import AnyChunk, ChunkType, ImageChunk, ImageType, TextChunk


@enum.unique
class RetrievalType(StrEnum):
    text = "text"
    image = "image"
    unknown = "unknown"


class ClassicIndexResultsPostprocessor(BaseRetriever):
    """
    Post-processor that extends documents returned by downstream retriever with field of RetrievalType,
    and expands an image chunks with pages to a list of text chunks from the corresponding page.

    * for text chunk: adds `RetrievalType.text`
    * for image chunk: the chunk gets replaced with the list of text chunk of this image's page,
      and each of them gets `RetrievalType.image` (in order to follow the logic of the original
      DIAL RAG where each retriever always return text chunks)

    """

    retriever: IndexRetriever = Field(repr=False)
    chunk_service: ChunkService = Field(repr=False)
    result_limit: int

    @classmethod
    def wrap(cls, retriever: IndexRetriever) -> Self:
        return cls(
            retriever=retriever,
            chunk_service=retriever.chunk_service,
            result_limit=retriever.top_k,
        )

    def _get_relevant_documents(self, query: str, *args, **kwargs) -> list[LangchainDocument]:
        raise NotImplementedError()

    async def _aget_relevant_documents(self, query: str, *args, **kwargs) -> list[LangchainDocument]:
        retrieved_docs = await self.retriever.ainvoke(query)
        result = []
        async for doc in self._postprocess_results(retrieved_docs):
            result.append(doc)
            if len(result) >= self.result_limit:
                break
        return result

    async def _postprocess_results(
        self, retrieved_docs: list[LangchainDocument]
    ) -> AsyncGenerator[LangchainDocument]:
        text_chunks_by_page = await self._collect_text_chunks_of_page_images(retrieved_docs)
        seen_pages: set[tuple[int, int]] = set()

        for doc in retrieved_docs:
            original_chunk: AnyChunk = doc.metadata["chunks"][0]
            page_key = (original_chunk.document_id, original_chunk.page_number)

            if original_chunk.chunk_type == ChunkType.image and page_key in text_chunks_by_page:
                # a page found by image search takes a single slot
                if page_key not in seen_pages:
                    seen_pages.add(page_key)
                    chunk = text_chunks_by_page[page_key][0]
                    yield LangchainDocument(
                        page_content=doc.page_content,
                        metadata={
                            "retrieval_type": RetrievalType.image,
                            "identity": chunk.get_identity(),
                            "chunks": [chunk],
                        },
                    )

            elif original_chunk.chunk_type == ChunkType.text:
                # for text chunks just add retrieval_type
                yield LangchainDocument(
                    page_content=doc.page_content,
                    metadata=dict(
                        retrieval_type=RetrievalType.text,
                        **doc.metadata,
                    ),
                )

            else:
                # if we are here, this case was not supported by classic DIAL RAG (for example, original_chunk
                # can be an image which is not "image of page"), so return this chunk "as is"
                yield LangchainDocument(
                    page_content=doc.page_content,
                    metadata=dict(
                        retrieval_type=RetrievalType.unknown,
                        **doc.metadata,
                    ),
                )

    async def _collect_text_chunks_of_page_images(
        self, retrieved_docs: list[LangchainDocument]
    ) -> dict[tuple[int, int], list[TextChunk]]:
        doc_pages = [
            (chunk.document_id, chunk.page_number)
            for doc in retrieved_docs
            for chunk in cast(list[AnyChunk], doc.metadata.get("chunks", []))
            if chunk.chunk_type == ChunkType.image and chunk.image_type == ImageType.page
        ]

        result: dict[tuple[int, int], list[TextChunk]] = {}

        for chunk in await self.chunk_service.get_chunks_by_pages(*doc_pages, chunk_type=ChunkType.text):
            assert isinstance(chunk, TextChunk)
            result.setdefault(
                (chunk.document_id, chunk.page_number),
                [],
            ).append(chunk)

        return result


class ClassicAggregatedResultPostprocessor(BaseRetriever):
    """
    Post-processor that repeats behavior of classic DIAL RAG retriever.

    * text chunks obtained by downstream index retrievers are returned "as is";
    * image chunks with page images are replaced by text chunks from corresponding pages;
    * final result contains text chunks, and some of them are complemented by images of corresponding page:
        * when selecting text chunks to attach page images, chunks with retrieval type `image`
          (i.e. those what was obtained by replacing page images) has higher priority over the other ones;
        * if final result contain several chunks from the same page, only one of them
          might be complemented by the image of the page;
        * adds up to `num_page_images_to_use` images to final result.

    Requires downstream instances of `IndexRetriever` to be wrapped with
    `ClassicIndexRetrieverResultsPostprocessor` retriever.

    """

    retriever: BaseRetriever = Field(repr=False)
    chunk_service: ChunkService = Field(repr=False)
    num_page_images_to_use: int

    def _get_relevant_documents(self, query: str, *args, **kwargs) -> list[LangchainDocument]:
        raise NotImplementedError()

    async def _aget_relevant_documents(self, query: str, *args, **kwargs) -> list[LangchainDocument]:
        retrieved_docs = await self.retriever.ainvoke(query)
        return [doc async for doc in self._postprocess_results(retrieved_docs)]

    async def _postprocess_results(
        self, retrieved_docs: list[LangchainDocument]
    ) -> AsyncGenerator[LangchainDocument]:
        image_by_page = await self._get_image_by_page(retrieved_docs)
        attached_images = set()

        for doc in retrieved_docs:
            original_chunk: AnyChunk = doc.metadata.get("chunks", [])[0]
            image_key = (original_chunk.document_id, original_chunk.page_number)

            if image_key not in attached_images and (page_image := image_by_page.get(image_key)):
                yield LangchainDocument(
                    page_content=doc.page_content,
                    metadata={
                        "retrieval_type": doc.metadata.get("retrieval_type"),
                        "identity": original_chunk.get_identity(),
                        "chunks": [original_chunk, page_image],
                    },
                )
                attached_images.add(image_key)
            else:
                yield doc

    async def _get_image_by_page(
        self, retrieved_docs: list[LangchainDocument]
    ) -> dict[tuple[int, int], ImageChunk]:
        required_pages: set[tuple[int, int]] = set()
        for document_id, page_number in self._collect_pages(retrieved_docs):
            required_pages.add((document_id, page_number))
            if len(required_pages) >= self.num_page_images_to_use:
                break

        result: dict[tuple[int, int], ImageChunk] = {}

        for chunk in await self.chunk_service.get_chunks_by_pages(
            *required_pages, chunk_type=ChunkType.image
        ):
            assert isinstance(chunk, ImageChunk)
            if chunk.image_type == ImageType.page:
                result[(chunk.document_id, chunk.page_number)] = chunk

        return result

    @staticmethod
    def _collect_pages(retrieved_docs: list[LangchainDocument]):
        for doc in retrieved_docs:
            chunk: AnyChunk = doc.metadata.get("chunks", [])[0]
            if doc.metadata.get("retrieval_type") == RetrievalType.image and chunk.page_number is not None:
                yield chunk.document_id, chunk.page_number
        for doc in retrieved_docs:
            chunk: AnyChunk = doc.metadata.get("chunks", [])[0]
            if doc.metadata.get("retrieval_type") == RetrievalType.text and chunk.page_number is not None:
                yield chunk.document_id, chunk.page_number


class ClassicRetrieverConfig(AbstractRetrieverConfig):
    num_page_images_to_use: int = Field(
        default=4,
        description="Sets maximum number of page images to pass to the model for the answer generation.",
    )


class ClassicRetriever[ConfigT: ClassicRetrieverConfig = ClassicRetrieverConfig](AbstractRetriever[ConfigT]):
    """
    Retriever that repeats behavior of classic DIAL RAG retriever.

    * text chunks obtained during indexes search are returned "as is";
    * image chunks with page images are replaced by text chunks from corresponding pages;
    * final result contains text chunks, and some of them are complemented by images of corresponding page:
        * when selecting text chunks to attach page images, chunks with retrieval type `image`
          (i.e. those what was obtained by replacing page images) has higher priority over the other ones;
        * if final result contain several chunks from the same page, only one of them
          might be complemented by the image of the page;
        * adds up to `num_page_images_to_use` images to final result;
    * image chunks with images of other types are returned "as is" without any transformation;
    * results from different indexes are combined in single list using rank fusion with equal weighting for all indexes.
    """

    @inject
    def __init__(self, config: ConfigT, channel: Channel, chunk_service: ChunkService):
        super().__init__(config, channel)
        self._chunk_service = chunk_service

    def _create_final_stage(self, intermediate_stages: list[RetrievalStage]) -> RetrievalStage:
        if len(intermediate_stages) == 1:
            return self._stage_factory(
                intermediate_stages[0].stage_name + " (classic)",
                ClassicAggregatedResultPostprocessor(
                    retriever=intermediate_stages[0].retriever,
                    chunk_service=self._chunk_service,
                    num_page_images_to_use=self.config.num_page_images_to_use,
                ),
            )

        return self._stage_factory(
            "Combined search (classic)",
            ClassicAggregatedResultPostprocessor(
                retriever=EnsembleRetriever(
                    retrievers=intermediate_stages,
                    weights=[1.0] * len(intermediate_stages),
                    id_key="identity",
                ),
                chunk_service=self._chunk_service,
                num_page_images_to_use=self.config.num_page_images_to_use,
            ),
        )

    def _create_intermediate_stage(
        self, index: ChunkIndex, documents: list[int] | None, top_k: int
    ) -> RetrievalStage:
        """Create :class:`RetrievalStage` for given index."""
        return self._stage_factory(
            index.display_name,
            ClassicIndexResultsPostprocessor.wrap(
                IndexRetriever(
                    index=index,
                    documents=documents,
                    top_k=top_k,
                    chunk_service=self._chunk_service,
                )
            ),
        )
