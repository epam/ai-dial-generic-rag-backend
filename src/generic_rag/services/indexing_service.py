import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterable

from injection import scoped

from generic_rag.channel import Channel
from generic_rag.scope import ScopeName
from generic_rag.types import AnyChunk, Document, ImageChunk, TextChunk
from generic_rag.utils.profile import log_execution_time

INDEXING_BATCH_SIZE = 1000  # todo: get from config

logger = logging.getLogger(__name__)


@scoped(ScopeName.channel)
class IndexingService:
    def __init__(self, channel: Channel):
        self._channel = channel

    @log_execution_time(logger)
    async def extract_chunks(self, document: Document) -> AsyncGenerator[AnyChunk]:
        """
        Extract chunks using document processors defined by given channel config.

        :param document: document to process
        """
        last_text_chunk_id = 0
        last_image_chunk_id = 0

        for parser in self._channel.document_parsers:
            async for chunk in await parser.extract_chunks(document):
                if isinstance(chunk, TextChunk):
                    last_text_chunk_id += 1
                    yield chunk.model_copy(update={"chunk_id": last_text_chunk_id})
                elif isinstance(chunk, ImageChunk):
                    last_image_chunk_id += 1
                    yield chunk.model_copy(update={"chunk_id": last_image_chunk_id})

        logger.info(f"extracted {last_text_chunk_id + last_image_chunk_id} chunk(s)")

    @log_execution_time(logger)
    async def index_chunks(self, chunks: AsyncIterable[AnyChunk], index_names: set[str] | None = None):
        """
        Index chunks with search indexes defined by given channel config.

        :param chunks: iterable of chunks to index
        :param index_names: list of index names to update
        """
        batch = []

        async def _process_batch():
            nonlocal batch
            tasks = []

            for index in await self._channel.get_indexes():
                if index_names and index.index_name not in index_names:
                    continue
                tasks.append(index.update(batch))

            if tasks:
                await asyncio.gather(*tasks)

            batch = []

        async for chunk in chunks:
            batch.append(chunk)
            if len(batch) >= INDEXING_BATCH_SIZE:
                await _process_batch()

        if batch:
            await _process_batch()
