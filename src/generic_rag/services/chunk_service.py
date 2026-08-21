import asyncio
import hashlib
import itertools
import logging
import mimetypes
import os
from collections.abc import AsyncIterable, Collection, Iterable
from typing import Literal, overload

from injection import scoped
from sqlalchemy import ScalarResult, delete, select, tuple_

from generic_rag.channel import Channel
from generic_rag.db.entities import ImageChunkEntity, TextChunkEntity
from generic_rag.db.session import get_current_session, transaction
from generic_rag.scope import ScopeName
from generic_rag.types import AnyChunk, ChunkRef, ChunkType, FileStorage, ImageChunk, TextChunk
from generic_rag.utils.profile import log_execution_time

_semaphore = asyncio.Semaphore(10)

logger = logging.getLogger(__name__)


@scoped(ScopeName.channel)
class ChunkService:
    """Service for managing storage of document's chunks."""

    def __init__(self, channel: Channel, file_storage: FileStorage):
        self._channel_key = channel.channel_key
        self._file_storage = file_storage

    async def add_chunks(self, chunks: AsyncIterable[AnyChunk], batch_size=50):
        """
        Add given chunks to the database.

        :param chunks: an iterable of Chunk instances to add
        :param batch_size: number of chunks in batch to store in database
        """
        batch: list[TextChunkEntity | ImageChunkEntity] = []

        async for chunk in chunks:
            batch.append(await self._create_entity(chunk))
            if len(batch) >= batch_size:
                await self._save_entities(batch)
                batch = []

        if batch:
            await self._save_entities(batch)

    @overload
    async def _create_entity(self, chunk: TextChunk) -> TextChunkEntity: ...

    @overload
    async def _create_entity(self, chunk: ImageChunk) -> ImageChunkEntity: ...

    async def _create_entity(self, chunk: TextChunk | ImageChunk) -> TextChunkEntity | ImageChunkEntity:
        """Create database entity for given chunk."""
        if isinstance(chunk, TextChunk):
            return TextChunkEntity(
                channel_key=self._channel_key,
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                page_number=chunk.page_number,
            )
        if isinstance(chunk, ImageChunk):
            image_url = await self._upload_image(chunk)
            return ImageChunkEntity(
                channel_key=self._channel_key,
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                image_type=chunk.image_type,
                image_url=image_url,
                mime_type=chunk.mime_type,
                page_number=chunk.page_number,
            )
        raise ValueError(f"unexpected argument: {chunk!r}")

    async def _upload_image(self, chunk: ImageChunk) -> str:
        """
        Upload content of image chunk to file storage.

        :param chunk: an image chunk to upload
        :return: url of the uploaded file
        """
        bucket = await self._file_storage.get_bucket()
        image_hash = hashlib.sha1(chunk.content).hexdigest()

        file_metadata = await self._file_storage.put_file(
            bucket=bucket,
            filepath=os.path.join(
                "images",
                str(chunk.document_id),
                str(chunk.chunk_id),
                image_hash + mimetypes.guess_extension(chunk.mime_type),
            ),
            content_type=chunk.mime_type,
            content=chunk.content,
        )
        return file_metadata.url

    @log_execution_time(logger)
    @transaction
    async def _save_entities(self, entities: list[TextChunkEntity | ImageChunkEntity]):
        get_current_session().add_all(entities)
        await get_current_session().flush()

    def get_chunks_by_document(self, document_id: int) -> AsyncIterable[AnyChunk]:
        """
        Load chunks of given document.

        All text chunks are yielded first, then all image chunks, and within each of these two
        groups the chunks come in indexing order. See `get_chunks_by_pages` for the assumption
        that `chunk_id` encodes that order.
        """

        @transaction
        async def _chunks_fetcher():
            # `chunk_id` is the position of the chunk in the indexing stream, so ordering by it
            # restores the order in which the parsers produced the chunks.
            text_entities: ScalarResult[TextChunkEntity] = await get_current_session().scalars(
                select(TextChunkEntity)
                .where(
                    TextChunkEntity.channel_key == self._channel_key,
                    TextChunkEntity.document_id == document_id,
                )
                .order_by(TextChunkEntity.chunk_id)
            )
            image_entities: ScalarResult[ImageChunkEntity] = await get_current_session().scalars(
                select(ImageChunkEntity)
                .where(
                    ImageChunkEntity.channel_key == self._channel_key,
                    ImageChunkEntity.document_id == document_id,
                )
                # ordered by `chunk_id` for the same reason as the text chunks above
                .order_by(ImageChunkEntity.chunk_id)
            )
            for entity in itertools.chain(text_entities, image_entities):
                yield await self._convert_entity(entity)

        return _chunks_fetcher()

    @transaction
    async def delete_chunks_by_document(self, document_id: int):
        """Delete chunks of given document."""

        async def _delete_file(url: str):
            async with _semaphore:
                await self._file_storage.delete_file(url)

        scalar_result: ScalarResult[str] = await get_current_session().scalars(
            select(ImageChunkEntity.image_url).where(
                ImageChunkEntity.channel_key == self._channel_key,
                ImageChunkEntity.document_id == document_id,
            )
        )
        image_files = scalar_result.all()

        await get_current_session().execute(
            delete(TextChunkEntity).where(
                TextChunkEntity.channel_key == self._channel_key,
                TextChunkEntity.document_id == document_id,
            )
        )
        await get_current_session().execute(
            delete(ImageChunkEntity).where(
                ImageChunkEntity.channel_key == self._channel_key,
                ImageChunkEntity.document_id == document_id,
            )
        )
        await get_current_session().flush()

        if tasks := [_delete_file(url) for url in image_files]:
            await asyncio.gather(*tasks)

    @transaction
    async def get_chunks_by_references(self, chunk_refs: Iterable[ChunkRef]) -> Collection[AnyChunk]:
        """Load chunks with given references."""
        text_chunk_ids: list[tuple[int, int]] = []
        image_chunk_ids: list[tuple[int, int]] = []
        tasks = []

        for ref in chunk_refs:
            match ref.chunk_type:
                case ChunkType.text:
                    text_chunk_ids.append((ref.document_id, ref.chunk_id))
                case ChunkType.image:
                    image_chunk_ids.append((ref.document_id, ref.chunk_id))

        if text_chunk_ids:
            text_entities: ScalarResult[TextChunkEntity] = await get_current_session().scalars(
                select(TextChunkEntity).where(
                    TextChunkEntity.channel_key == self._channel_key,
                    tuple_(TextChunkEntity.document_id, TextChunkEntity.chunk_id).in_(text_chunk_ids),
                )
            )
            tasks.extend(self._convert_entity(entity) for entity in text_entities.all())

        if image_chunk_ids:
            image_entities: ScalarResult[ImageChunkEntity] = await get_current_session().scalars(
                select(ImageChunkEntity).where(
                    ImageChunkEntity.channel_key == self._channel_key,
                    tuple_(ImageChunkEntity.document_id, ImageChunkEntity.chunk_id).in_(image_chunk_ids),
                )
            )
            tasks.extend(self._convert_entity(entity) for entity in image_entities.all())

        return await asyncio.gather(*tasks) if tasks else []

    @overload
    async def get_chunks_by_pages(
        self, *doc_pages: tuple[int, int], chunk_type: Literal[ChunkType.text]
    ) -> Collection[TextChunk]: ...

    @overload
    async def get_chunks_by_pages(
        self, *doc_pages: tuple[int, int], chunk_type: Literal[ChunkType.image]
    ) -> Collection[ImageChunk]: ...

    @overload
    async def get_chunks_by_pages(
        self, *doc_pages: tuple[int, int], chunk_type: None = None
    ) -> Collection[AnyChunk]: ...

    @transaction
    async def get_chunks_by_pages(
        self, *doc_pages: tuple[int, int], chunk_type: ChunkType | None = None
    ) -> Collection[TextChunk | ImageChunk]:
        """
        Return chunks for given documents pages.

        Chunks are returned grouped by document, and within a document in the order in which they
        were indexed. Callers that concatenate the chunks of a page (for example to rebuild the
        page text) depend on this ordering.

        The ordering assumes that `chunk_id` is the sequential number of the chunk in the indexing
        stream. `IndexingService._extract_chunks` guarantees this: it overrides the id assigned by
        the parser with its own counter, which it increments by one for every chunk, in the order
        the parsers yield them. Text and image chunks are numbered by two independent counters, so
        the ordering is meaningful within each chunk type but says nothing about how chunks of
        different types relate to each other. Note also that indexing order equals the document's
        reading order only as long as parsers emit their chunks sequentially.

        :param doc_pages: pairs of (document_id, page_number) describing required pages
        :param chunk_type: defines what types of chunks to return
        """
        if not doc_pages:
            return []

        tasks = []

        if not chunk_type or chunk_type == ChunkType.text:
            # `chunk_id` is the position of the chunk in the indexing stream, so ordering by it
            # restores the order in which the parsers produced the chunks.
            text_entities: ScalarResult[TextChunkEntity] = await get_current_session().scalars(
                select(TextChunkEntity)
                .where(
                    TextChunkEntity.channel_key == self._channel_key,
                    tuple_(TextChunkEntity.document_id, TextChunkEntity.page_number).in_(doc_pages),
                )
                .order_by(TextChunkEntity.document_id, TextChunkEntity.chunk_id)
            )
            tasks.extend(self._convert_entity(entity) for entity in text_entities.all())

        if not chunk_type or chunk_type == ChunkType.image:
            image_entities: ScalarResult[ImageChunkEntity] = await get_current_session().scalars(
                select(ImageChunkEntity)
                .where(
                    ImageChunkEntity.channel_key == self._channel_key,
                    tuple_(ImageChunkEntity.document_id, ImageChunkEntity.page_number).in_(doc_pages),
                )
                # ordered by `chunk_id` for the same reason as the text chunks above
                .order_by(ImageChunkEntity.document_id, ImageChunkEntity.chunk_id)
            )
            tasks.extend(self._convert_entity(entity) for entity in image_entities.all())

        return await asyncio.gather(*tasks)

    @overload
    async def _convert_entity(self, entity: TextChunkEntity) -> TextChunk: ...

    @overload
    async def _convert_entity(self, entity: ImageChunkEntity) -> ImageChunk: ...

    async def _convert_entity(self, entity: TextChunkEntity | ImageChunkEntity) -> TextChunk | ImageChunk:
        """Convert given entity into a chunk."""
        if isinstance(entity, TextChunkEntity):
            return TextChunk(
                document_id=entity.document_id,
                chunk_id=entity.chunk_id,
                page_number=entity.page_number,
                text=entity.text,
            )
        if isinstance(entity, ImageChunkEntity):
            async with _semaphore:
                image_content = b"".join([
                    data async for data in await self._file_storage.download_file(entity.image_url)
                ])
            return ImageChunk(
                document_id=entity.document_id,
                chunk_id=entity.chunk_id,
                page_number=entity.page_number,
                image_type=entity.image_type,
                mime_type=entity.mime_type,
                content=image_content,
            )
        raise ValueError(f"unexpected argument: {entity!r}")
