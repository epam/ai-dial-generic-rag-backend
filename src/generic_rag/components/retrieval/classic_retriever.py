import enum
from collections.abc import AsyncGenerator, Sequence
from enum import StrEnum

from injection import inject
from pydantic import Field

from generic_rag.channel import Channel
from generic_rag.components.retrieval.abstract_retriever import AbstractRetriever, AbstractRetrieverConfig
from generic_rag.components.search_index import ChunkIndex
from generic_rag.services.chunk_service import ChunkService
from generic_rag.types import AnyChunk, ChunkType, ImageChunk, ImageType, RetrievedDocument, TextChunk
from generic_rag.utils.ranking import rank_fusion


@enum.unique
class RetrievalType(StrEnum):
    text = "text"
    image = "image"
    unknown = "unknown"


class ClassicIndexResultsPostprocessor:
    """
    Post-processor that extends documents returned by downstream retriever with field of RetrievalType,
    and expands an image chunks with pages to a list of text chunks from the corresponding page.

    * for text chunk: adds `RetrievalType.text`
    * for image chunk: the chunk gets replaced with the list of text chunk of this image's page,
      and each of them gets `RetrievalType.image` (in order to follow the logic of the original
      DIAL RAG where each retriever always return text chunks)

    """

    @inject
    def __init__(self, result_limit: int, chunk_service: ChunkService = NotImplemented):
        self._result_limit = result_limit
        self._chunk_service = chunk_service

    async def invoke(self, retrieved_docs: Sequence[RetrievedDocument]) -> Sequence[RetrievedDocument]:
        result = []
        async for doc in self._postprocess_results(retrieved_docs):
            result.append(doc)
            if len(result) >= self._result_limit:
                break
        return result

    async def _postprocess_results(
        self, retrieved_docs: Sequence[RetrievedDocument]
    ) -> AsyncGenerator[RetrievedDocument]:
        text_chunks_by_page = await self._collect_text_chunks_of_page_images(retrieved_docs)

        for doc in retrieved_docs:
            assert len(doc.chunks) == 1
            original_chunk: AnyChunk = doc.chunks[0]
            page_key = (original_chunk.document_id, original_chunk.page_number)

            if original_chunk.chunk_type == ChunkType.image and page_key in text_chunks_by_page:
                # instead of returning original image chunk with page image we return
                # text chunks from the page where this image chunk is originated
                for chunk in text_chunks_by_page.get(page_key, []):
                    yield doc.model_copy(
                        update={
                            "chunks": [chunk],
                            "retrieval_type": RetrievalType.image,
                        }
                    )

            elif original_chunk.chunk_type == ChunkType.text:
                # for text chunks just add retrieval_type
                yield doc.model_copy(update={"retrieval_type": RetrievalType.text})

            else:
                # if we are here, this case was not supported by classic DIAL RAG (for example, original_chunk
                # can be an image which is not "image of page"), so return this chunk "as is"
                yield doc.model_copy(update={"retrieval_type": RetrievalType.unknown})

    async def _collect_text_chunks_of_page_images(
        self, retrieved_docs: Sequence[RetrievedDocument]
    ) -> dict[tuple[int, int], list[TextChunk]]:
        doc_pages = [
            (chunk.document_id, chunk.page_number)
            for doc in retrieved_docs
            for chunk in doc.chunks
            if isinstance(chunk, ImageChunk) and chunk.image_type == ImageType.page
        ]

        result: dict[tuple[int, int], list[TextChunk]] = {}

        for chunk in await self._chunk_service.get_chunks_by_pages(*doc_pages, chunk_type=ChunkType.text):
            assert isinstance(chunk, TextChunk)
            result.setdefault(
                (chunk.document_id, chunk.page_number),
                [],
            ).append(chunk)

        return result


class ClassicAggregatedResultPostprocessor:
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

    @inject
    def __init__(self, num_page_images_to_use: int, chunk_service: ChunkService = NotImplemented):
        self._num_page_images_to_use = num_page_images_to_use
        self._chunk_service = chunk_service

    async def invoke(self, retrieved_docs: Sequence[RetrievedDocument]) -> Sequence[RetrievedDocument]:
        return [doc async for doc in self._postprocess_results(retrieved_docs)]

    async def _postprocess_results(
        self, retrieved_docs: Sequence[RetrievedDocument]
    ) -> AsyncGenerator[RetrievedDocument]:
        image_by_page = await self._get_image_by_page(retrieved_docs)
        attached_images = set()

        for doc in retrieved_docs:
            assert len(doc.chunks) == 1
            assert isinstance(doc.model_extra, dict)

            original_chunk: AnyChunk = doc.chunks[0]
            image_key = (original_chunk.document_id, original_chunk.page_number)

            if image_key not in attached_images and (page_image := image_by_page.get(image_key)):
                yield doc.model_copy(update={"chunks": [original_chunk, page_image]})
                attached_images.add(image_key)
            else:
                yield doc

    async def _get_image_by_page(
        self, retrieved_docs: Sequence[RetrievedDocument]
    ) -> dict[tuple[int, int], ImageChunk]:
        required_pages: set[tuple[int, int]] = set()
        for document_id, page_number in self._collect_pages(retrieved_docs):
            required_pages.add((document_id, page_number))
            if len(required_pages) >= self._num_page_images_to_use:
                break

        result: dict[tuple[int, int], ImageChunk] = {}

        for chunk in await self._chunk_service.get_chunks_by_pages(
            *required_pages, chunk_type=ChunkType.image
        ):
            assert isinstance(chunk, ImageChunk)
            if chunk.image_type == ImageType.page:
                result[(chunk.document_id, chunk.page_number)] = chunk

        return result

    @staticmethod
    def _collect_pages(retrieved_docs: Sequence[RetrievedDocument]):
        for doc in retrieved_docs:
            chunk: AnyChunk = doc.chunks[0]
            if doc.model_extra.get("retrieval_type") == RetrievalType.image:
                yield chunk.document_id, chunk.page_number
        for doc in retrieved_docs:
            chunk: AnyChunk = doc.chunks[0]
            if doc.model_extra.get("retrieval_type") == RetrievalType.text:
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

    async def _index_search(
        self, query: str, index: ChunkIndex, top_k: int, documents: list[int] | None = None
    ) -> Sequence[RetrievedDocument]:
        return await ClassicIndexResultsPostprocessor(top_k).invoke(
            await super()._index_search(query, index, top_k, documents)
        )

    async def _get_combined_results(
        self, doc_lists: list[Sequence[RetrievedDocument]]
    ) -> Sequence[RetrievedDocument]:
        return await ClassicAggregatedResultPostprocessor(
            num_page_images_to_use=self.config.num_page_images_to_use
        ).invoke(
            await rank_fusion(
                doc_lists,
                key=lambda doc: doc.chunks[0].get_identity(),
            )
        )
