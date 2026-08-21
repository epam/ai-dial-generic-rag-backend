import logging
from asyncio import TaskGroup
from collections.abc import AsyncGenerator, AsyncIterable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from injection import inject, scoped

from generic_rag.channel import Channel
from generic_rag.scope import ScopeName
from generic_rag.services.chunk_service import ChunkService
from generic_rag.services.document_service import DocumentService
from generic_rag.types import AnyChunk, Document, DocumentStatus, ImageChunk, TextChunk
from generic_rag.utils.profile import log_execution_time

INDEXING_BATCH_SIZE = 1000  # todo: get from config

logger = logging.getLogger(__name__)


class DocumentStatusUpdateHelper:
    """Helper class to run document processing with automatic status update upon the processing."""

    @inject
    def __init__(self, document_id: int, document_service: DocumentService = NotImplemented):
        self._document_id = document_id
        self._document_service = document_service

    def begin(self, *, started: DocumentStatus, finished: DocumentStatus) -> AbstractAsyncContextManager:
        """
        Get context manager automatic status update.

        :param started: a status that should be set in the beginning
        :param finished: a status that should be set if the processing has finished without errors
        """

        @asynccontextmanager
        async def _helper():
            await self._document_service.set_document_status(self._document_id, started)
            try:
                yield
            except BaseException as e:
                logger.error(str(e))
                await self._document_service.set_document_status(self._document_id, DocumentStatus.error)
                raise e
            else:
                await self._document_service.set_document_status(self._document_id, finished)

        return _helper()


@scoped(ScopeName.channel)
class IndexingService:
    def __init__(self, channel: Channel = NotImplemented, chunk_service: ChunkService = NotImplemented):
        self._channel = channel
        self._chunk_service = chunk_service

    @log_execution_time(logger)
    async def index_document(
        self, document: Document, *, index_names: set[str] | None = None, force: bool = False
    ):
        """
        Index or reindex given document.

        :param document: document to index
        :param index_names: names of indexes to update (if not defined - all indexes will be updated)
        :param force: perform whole process, including document re-processing and rebuilding of all indexes;
          it not set, document processing will be performed only if the document wasn't processed yet
        """
        status_helper = DocumentStatusUpdateHelper(document.id)

        if force or document.status not in [DocumentStatus.processed, DocumentStatus.ready]:
            async with status_helper.begin(
                started=DocumentStatus.processing, finished=DocumentStatus.processed
            ):
                await self._chunk_service.delete_chunks_by_document(document.id)
                await self._chunk_service.add_chunks(
                    self._extract_chunks(document),
                )
                index_names = None  # trigger rebuilding of all indexes

        async with status_helper.begin(started=DocumentStatus.indexing, finished=DocumentStatus.ready):
            async with TaskGroup() as task_group:
                for idx in await self._channel.get_indexes():
                    if index_names is None or idx.index_name in index_names:
                        task_group.create_task(idx.storage.remove(document.id))

            await self._index_chunks(
                self._chunk_service.get_chunks_by_document(
                    document.id,
                ),
                index_names=index_names,
            )

    @log_execution_time(logger)
    async def _extract_chunks(self, document: Document) -> AsyncGenerator[AnyChunk]:
        """
        Extract chunks using document processors defined by given channel config.

        :param document: document to process
        """
        last_text_chunk_id = 0
        last_image_chunk_id = 0

        for parser in self._channel.document_parsers:
            if document.mime_type not in parser.supported_mime_types:
                continue

            async for chunk in await parser.extract_chunks(document):
                if isinstance(chunk, TextChunk):
                    last_text_chunk_id += 1
                    yield chunk.model_copy(update={"chunk_id": last_text_chunk_id})
                elif isinstance(chunk, ImageChunk):
                    last_image_chunk_id += 1
                    yield chunk.model_copy(update={"chunk_id": last_image_chunk_id})

        logger.info(f"extracted {last_text_chunk_id + last_image_chunk_id} chunk(s)")

    @log_execution_time(logger)
    async def _index_chunks(self, chunks: AsyncIterable[AnyChunk], index_names: set[str] | None = None):
        """
        Index chunks with search indexes defined by given channel config.

        :param chunks: iterable of chunks to index
        :param index_names: list of index names to update
        """
        batch = []
        batch_number = 0

        async def _process_batch():
            logger.info(f"processing batch: #{batch_number}")

            async with TaskGroup() as task_group:
                for index in await self._channel.get_indexes():
                    if index_names is None or index.index_name in index_names:
                        task_group.create_task(index.add(batch))

        async for chunk in chunks:
            batch.append(chunk)
            if len(batch) >= INDEXING_BATCH_SIZE:
                await _process_batch()
                batch = []
                batch_number += 1

        if batch:
            await _process_batch()
