from collections.abc import Collection, Sequence
from typing import Annotated

from injection import scoped
from pydantic import BaseModel, Field
from sqlalchemy import INTEGER, ColumnElement, Select, func, select, union_all

from generic_rag.channel import Channel
from generic_rag.db.entities import DocumentEntity, ImageChunkEntity, TextChunkEntity
from generic_rag.db.session import get_current_session, transaction
from generic_rag.scope import ScopeName
from generic_rag.types import DocumentStatus


class DocumentStats(BaseModel):
    """Document statistics."""

    document_id: Annotated[int, Field(..., description="id of the document")]
    number_of_pages: Annotated[int | None, Field(description="Number of document pages", ge=0)] = None


class PageChunkStats(BaseModel):
    """Chunk statistics of a single document page."""

    page_number: Annotated[int, Field(description="Number of the page (1 based)", ge=1)]
    text_chunks: Annotated[int, Field(description="Number of text chunks extracted from the page", ge=0)]
    image_chunks: Annotated[int, Field(description="Number of image chunks extracted from the page", ge=0)]
    text_characters: Annotated[
        int, Field(description="Total number of characters across the page's text chunks", ge=0)
    ]


class DocumentChunkStats(BaseModel):
    """Chunk statistics of a single document."""

    document_id: Annotated[int, Field(description="id of the document")]
    display_name: Annotated[str, Field(description="User-facing name of the document")]
    status: Annotated[DocumentStatus, Field(description="Processing status of the document")]
    number_of_pages: Annotated[int, Field(description="Number of pages that produced any chunk", ge=0)]
    text_chunks: Annotated[int, Field(description="Number of text chunks in the document", ge=0)]
    image_chunks: Annotated[int, Field(description="Number of image chunks in the document", ge=0)]
    text_characters: Annotated[
        int, Field(description="Total number of characters across the document's text chunks", ge=0)
    ]
    pages: Annotated[
        list[PageChunkStats] | None,
        Field(description="Per-page breakdown, present only when the document was requested by id"),
    ] = None


class ChannelChunkStats(BaseModel):
    """Chunk statistics of a whole channel."""

    documents: Annotated[int, Field(description="Number of documents in the channel", ge=0)]
    documents_without_text: Annotated[
        int,
        Field(
            description=(
                "Number of documents that produced no text chunk at all. A document that is `ready` "
                "and still counted here was parsed without any text being extracted."
            ),
            ge=0,
        ),
    ]
    text_chunks: Annotated[int, Field(description="Number of text chunks in the channel", ge=0)]
    image_chunks: Annotated[int, Field(description="Number of image chunks in the channel", ge=0)]
    text_characters: Annotated[
        int, Field(description="Total number of characters across the channel's text chunks", ge=0)
    ]
    results: Annotated[list[DocumentChunkStats], Field(description="Per-document statistics")]


@scoped(ScopeName.channel)
class DocumentStatsService:
    """Service for getting documents statistics."""

    def __init__(self, channel: Channel):
        self._channel_key = channel.channel_key

    @transaction
    async def get_document_stats(self, *document_ids: int) -> Sequence[DocumentStats]:
        """
        Get statistics for given documents.

        The highest page number is taken over the union of both chunk tables rather than over a
        join of them. Joining both to `documents` pairs every text chunk with every image chunk of
        the same document, so the rows scanned grow as their product: a document with 500 text and
        50 image chunks materialises 25 000 rows to answer with one number.
        """
        if not document_ids:
            return []

        ids = set(document_ids)
        pages = union_all(*[
            select(
                entity.document_id.label("document_id"),
                func.cast(entity.metadata_["page_number"], INTEGER).label("page_number"),
            ).where(entity.channel_key == self._channel_key, entity.document_id.in_(ids))
            for entity in (TextChunkEntity, ImageChunkEntity)
        ]).subquery()

        max_page = (
            select(
                pages.c.document_id,
                func.max(pages.c.page_number).label("number_of_pages"),
            )
            .group_by(pages.c.document_id)
            .subquery()
        )

        result = await get_current_session().execute(
            select(DocumentEntity.document_id, max_page.c.number_of_pages)
            .outerjoin(max_page, max_page.c.document_id == DocumentEntity.document_id)
            .where(
                DocumentEntity.channel_key == self._channel_key,
                DocumentEntity.document_id.in_(ids),
            )
        )
        return [
            DocumentStats(
                document_id=id_,
                number_of_pages=number_of_pages,
            )
            for id_, number_of_pages in result.all()
        ]

    @transaction
    async def get_channel_chunk_stats(
        self, document_ids: Collection[int] | None = None, *, include_pages: bool = False
    ) -> ChannelChunkStats:
        """
        Return chunk statistics for the channel, or for the given documents only.

        Text and image chunks are counted with separate queries on purpose. Joining both chunk
        tables in one statement multiplies their rows, which is harmless for the `max()` that
        `get_document_stats` needs but would inflate every count here.

        :param document_ids: restrict the report to these documents; all documents when omitted
        :param include_pages: fill in the per-page breakdown of each document
        """
        text_by_page = await self._count_by_page(
            TextChunkEntity,
            document_ids,
            characters=func.coalesce(func.sum(func.length(TextChunkEntity.text)), 0),
        )
        image_by_page = await self._count_by_page(ImageChunkEntity, document_ids)

        documents = await self._get_documents(document_ids)

        results: list[DocumentChunkStats] = []
        for document_id, display_name, status in documents:
            text_pages = text_by_page.get(document_id, {})
            image_pages = image_by_page.get(document_id, {})

            pages = [
                PageChunkStats(
                    page_number=page_number,
                    text_chunks=text_pages.get(page_number, (0, 0))[0],
                    image_chunks=image_pages.get(page_number, (0, 0))[0],
                    text_characters=text_pages.get(page_number, (0, 0))[1],
                )
                for page_number in sorted(set(text_pages) | set(image_pages))
            ]

            results.append(
                DocumentChunkStats(
                    document_id=document_id,
                    display_name=display_name,
                    status=status,
                    number_of_pages=len(pages),
                    text_chunks=sum(page.text_chunks for page in pages),
                    image_chunks=sum(page.image_chunks for page in pages),
                    text_characters=sum(page.text_characters for page in pages),
                    pages=pages if include_pages else None,
                )
            )

        return ChannelChunkStats(
            documents=len(results),
            documents_without_text=sum(1 for document in results if document.text_chunks == 0),
            text_chunks=sum(document.text_chunks for document in results),
            image_chunks=sum(document.image_chunks for document in results),
            text_characters=sum(document.text_characters for document in results),
            results=results,
        )

    async def _get_documents(
        self, document_ids: Collection[int] | None
    ) -> Sequence[tuple[int, str, DocumentStatus]]:
        """Return the (id, display name, status) of the documents the report covers."""
        query = select(DocumentEntity.document_id, DocumentEntity.display_name, DocumentEntity.status).where(
            DocumentEntity.channel_key == self._channel_key
        )
        query = self._restrict_to_documents(query, DocumentEntity.document_id, document_ids)
        cursor = await get_current_session().execute(query.order_by(DocumentEntity.document_id))
        return cursor.all()  # type: ignore[return-value]

    async def _count_by_page(
        self,
        entity: type[TextChunkEntity] | type[ImageChunkEntity],
        document_ids: Collection[int] | None,
        characters: ColumnElement[int] | None = None,
    ) -> dict[int, dict[int, tuple[int, int]]]:
        """
        Count chunks of one kind per document and page.

        :returns: `{document_id: {page_number: (chunk_count, character_count)}}`
        """
        page_number = func.cast(entity.metadata_["page_number"], INTEGER).label("page_number")
        query = (
            select(
                entity.document_id,
                page_number,
                func.count().label("chunks"),
                (characters if characters is not None else func.cast(0, INTEGER)).label("characters"),
            )
            .where(entity.channel_key == self._channel_key)
            .group_by(entity.document_id, page_number)
        )
        query = self._restrict_to_documents(query, entity.document_id, document_ids)

        result: dict[int, dict[int, tuple[int, int]]] = {}
        for document_id, page, chunks, characters_count in await get_current_session().execute(query):
            if page is None:
                # a chunk whose metadata carries no page number cannot be attributed to a page
                continue
            result.setdefault(document_id, {})[page] = (chunks, characters_count)
        return result

    @staticmethod
    def _restrict_to_documents[T](
        query: Select[T], column: ColumnElement[int], document_ids: Collection[int] | None
    ) -> Select[T]:
        """Narrow a query to the given documents, leaving it untouched when none are given."""
        if document_ids is None:
            return query
        return query.where(column.in_(set(document_ids)))
