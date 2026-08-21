import io
import logging
from collections.abc import AsyncGenerator, AsyncIterable
from functools import cached_property

from pydantic import BaseModel, Field, model_validator
from unstructured.partition.auto import partition

from generic_rag.types import Document, DocumentParser, TextChunk
from generic_rag.utils.profile import log_execution_time

logger = logging.getLogger(__name__)


class UnstructuredParserConfig(BaseModel):
    chunk_size: int = Field(default=1000, description="the chunk size for unstructured document loader")
    combine_text_under_n_chars: int = Field(
        default=100, description="combine small chunks until reaching this many characters"
    )

    @model_validator(mode="after")
    def check_combine_within_chunk_size(self) -> "UnstructuredParserConfig":
        if self.combine_text_under_n_chars > self.chunk_size:
            raise ValueError(
                f"combine_text_under_n_chars ({self.combine_text_under_n_chars}) "
                f"must not exceed chunk_size ({self.chunk_size})"
            )
        return self


class UnstructuredParser(DocumentParser[UnstructuredParserConfig]):
    """Parser that extracts text chunks using `unstructured` library."""

    @cached_property
    def supported_mime_types(self) -> frozenset[str]:
        return frozenset({
            "application/pdf",
            "text/markdown",
            "text/plain",
        })

    async def extract_chunks(self, document: Document) -> AsyncIterable[TextChunk]:
        return self._extract_chunks_gen(document)

    @log_execution_time(logger)
    async def _extract_chunks_gen(self, document: Document) -> AsyncGenerator[TextChunk]:
        assert document.mime_type in self.supported_mime_types

        elements = partition(
            file=io.BytesIO(await document.get_content()),
            content_type=document.mime_type,
            metadata_filename=document.display_name,
            strategy="fast",
            chunking_strategy="by_title",
            multipage_sections=False,
            combine_text_under_n_chars=self.config.combine_text_under_n_chars,
            new_after_n_chars=self.config.chunk_size,
            max_characters=self.config.chunk_size,
        )

        for i, element in enumerate(elements, start=1):
            yield TextChunk(
                document_id=document.id,
                chunk_id=i,
                page_number=element.metadata.page_number or 0,
                text=element.text,
            )
