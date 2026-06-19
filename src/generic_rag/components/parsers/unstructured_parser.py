import io
import logging
from collections.abc import AsyncGenerator, AsyncIterable

from pydantic import BaseModel, Field
from unstructured.partition.auto import partition

from generic_rag.types import Document, DocumentParser, TextChunk
from generic_rag.utils.profile import log_execution_time

logger = logging.getLogger(__name__)


class UnstructuredParserConfig(BaseModel):
    chunk_size: int = Field(
        default=1000,
        description="the chunk size for unstructured document loader"
    )


class UnstructuredParser(DocumentParser[UnstructuredParserConfig]):
    """ Parser that extracts text chunks using `unstructured` library. """

    async def extract_chunks(self, document: Document) -> AsyncIterable[TextChunk]:
        return self._extract_chunks_gen(document)

    @log_execution_time(logger)
    async def _extract_chunks_gen(self, document: Document) -> AsyncGenerator[TextChunk]:
        document_content = await document.get_content()

        elements = partition(
            file=io.BytesIO(document_content),
            content_type=document.mime_type,
            metadata_filename=document.display_name,
            strategy="fast",
            chunking_strategy="by_title",
            multipage_sections=False,
            combine_text_under_n_chars=0,
            new_after_n_chars=self.config.chunk_size,
            max_characters=self.config.chunk_size,
        )

        for i, element in enumerate(elements, start=1):
            yield TextChunk(
                document_id=document.id,
                chunk_id=i,
                page_number=element.metadata.page_number,
                text=element.text,
            )
