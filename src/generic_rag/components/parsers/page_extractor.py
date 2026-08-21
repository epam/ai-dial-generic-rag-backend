import io
import logging
from collections.abc import AsyncGenerator, AsyncIterable
from functools import cached_property

import pdfplumber
from pdfplumber.page import Page
from PIL.Image import Image
from pydantic import BaseModel, Field

from generic_rag.types import Document, DocumentParser, ImageChunk, ImageType
from generic_rag.utils.profile import log_execution_time

logger = logging.getLogger(__name__)


class PageExtractorConfig(BaseModel):
    image_size: int = Field(
        default=1536,
        description="maximum size of extracted image",
    )


class PageExtractor(DocumentParser[PageExtractorConfig]):
    """Parser that extracts images of document pages.

    Applies to PDF documents only. A channel can be configured with several parsers and hold
    documents of several types, so anything else is skipped rather than failed: the other
    parsers still have chunks to contribute, and one document that has no pages to render must
    not leave the whole document unindexed.
    """

    @cached_property
    def supported_mime_types(self) -> frozenset[str]:
        return frozenset({"application/pdf"})

    async def extract_chunks(self, document: Document) -> AsyncIterable[ImageChunk]:
        return self._extract_chunks_gen(document)

    @log_execution_time(logger)
    async def _extract_chunks_gen(self, document: Document) -> AsyncGenerator[ImageChunk]:
        assert document.mime_type in self.supported_mime_types

        with pdfplumber.open(io.BytesIO(await document.get_content())) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                logger.info(f"processing page {page_number}...")

                image = self._get_page_image(page, scaled_size=self.config.image_size)

                with io.BytesIO() as fp:
                    image.save(fp, format="png")
                    image_content = fp.getvalue()

                yield ImageChunk(
                    document_id=document.id,
                    chunk_id=page_number,
                    page_number=page_number,
                    image_type=ImageType.page,
                    mime_type="image/png",
                    content=image_content,
                )

    @staticmethod
    def _get_page_image(page: Page, scaled_size: int | None) -> Image:
        width = None
        height = None
        if page.width > page.height:
            width = scaled_size
        else:
            height = scaled_size
        return page.to_image(width=width, height=height).original
