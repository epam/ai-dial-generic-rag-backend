import asyncio
import inspect
import io
import logging
import os.path
from collections.abc import AsyncGenerator, AsyncIterable
from functools import cached_property

import unstructured.partition.pdf
import wrapt
from pydantic import Field, model_validator
from unstructured.documents.elements import Element
from unstructured.partition.auto import partition
from unstructured.partition.utils.constants import PartitionStrategy

from generic_rag.types import ChunkMetadata, Document, DocumentParser, DocumentParserConfig, TextChunk
from generic_rag.utils.profile import log_execution_time

logger = logging.getLogger(__name__)


# noinspection unused-parameter
def _is_pdf_too_complex_wrapper(wrapped, instance, args, kwargs):
    for frame_info in inspect.stack():
        if (
            frame_info.filename == unstructured.partition.pdf.__file__
            and frame_info.function == "partition_pdf"
            and frame_info.frame.f_locals.get("strategy") == PartitionStrategy.FAST
        ):
            # Disable PDF complexity check enforced by unstructured.
            #
            # Starting from unstructured 0.22.4 it enforces invocation of `is_pdf_too_complex`
            # for every single file no matter what partition strategy was specified. It's totally makes
            # sense when use `auto` strategy, but with `fast` strategy we don't want to use OCR and
            # unstructured actually does not perform any "fallback to hi_res" as it states in logs
            # nor throwing an error. Instead, it just skips text extraction and does nothing.
            # So by disabling this check we return back the old behavior which was changed in 0.22.4.
            #
            # See:
            #   https://github.com/Unstructured-IO/unstructured/pull/4268
            #   https://github.com/Unstructured-IO/unstructured/releases/tag/0.22.4
            #
            return False

    return wrapped(*args, **kwargs)


wrapt.wrap_function_wrapper(unstructured.partition.pdf, "is_pdf_too_complex", _is_pdf_too_complex_wrapper)


class UnstructuredParserConfig(DocumentParserConfig):
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
    """Uses `unstructured` library to extract chunks."""

    @cached_property
    def supported_mime_types(self) -> frozenset[str]:
        return frozenset({
            "application/pdf",
            "text/markdown",
            "text/plain",
        })

    def _extract_chunks(self, document: Document) -> AsyncIterable[TextChunk]:
        return self._extract_chunks_gen(document)

    @log_execution_time(logger)
    async def _extract_chunks_gen(self, document: Document) -> AsyncGenerator[TextChunk]:
        assert document.mime_type in self.supported_mime_types

        # check `unstructured.chunking.dispatch.add_chunking_strategy` for available options
        chunking_kwargs = {
            "chunking_strategy": "by_title",
            "combine_text_under_n_chars": self.config.combine_text_under_n_chars,
            "new_after_n_chars": self.config.chunk_size,
            "max_characters": self.config.chunk_size,
        }

        elements: list[Element] = await asyncio.to_thread(
            partition,
            file=io.BytesIO(await document.get_content()),
            content_type=document.mime_type,
            strategy=PartitionStrategy.FAST,
            metadata_filename=os.path.basename(document.display_name),
            **chunking_kwargs,
        )

        for i, element in enumerate(elements, start=1):
            yield TextChunk(
                document_id=document.id,
                chunk_id=i,
                text=element.text,
                metadata=ChunkMetadata(
                    page_number=element.metadata.page_number or 0,
                ),
            )
