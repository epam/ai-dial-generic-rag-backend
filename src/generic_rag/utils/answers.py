import asyncio
import io
import itertools
import logging
import time
import traceback
import uuid
from contextlib import suppress
from types import TracebackType
from typing import Any, Self

from aidial_sdk.chat_completion import Attachment, Choice, Stage
from aidial_sdk.chat_completion.chunks import ArbitraryChunk
from aidial_sdk.utils.json import remove_nones
from datauri import DataURI
from injection import inject
from opentelemetry.trace import INVALID_SPAN, INVALID_SPAN_CONTEXT, get_current_span
from PIL import Image
from PIL.Image import Resampling

from generic_rag.app.settings import ChatSettings
from generic_rag.services.document_service import DocumentService
from generic_rag.types import (
    Answer,
    AnswerStage,
    FileStorage,
    ImageChunk,
    ImageType,
    RetrievedDocument,
    TextChunk,
)

logger = logging.getLogger(__name__)

CITATION_TAG = "cit"


class NoopStage(AnswerStage):
    """Stage that does nothing."""

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, tb: TracebackType | None
    ): ...

    def append_content(self, content: str): ...

    async def add_citation(self, doc: RetrievedDocument): ...


class PlainAnswer(Answer):
    """Answer implementation which accumulates the content in plain string."""

    def __init__(self):
        self._content: str = ""
        self._has_references = False

    @property
    def content(self):
        return self._content

    @property
    def has_references(self) -> bool:
        return self._has_references

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, tb: TracebackType | None
    ): ...

    def create_stage(self, name: str, *, debug: bool = False, timed: bool = True) -> AnswerStage:
        return NoopStage()

    def append_content(self, content: str):
        self._content += content

    async def add_citation(self, doc: RetrievedDocument):
        # Name both parts, so a consumer reading the text alone can tell the document from the page.
        # This is the form `get_pages` emits, so the whole MCP server cites documents the same way.
        self.append_content(f"[Document {doc.source_id}, Page {doc.source_page_number}]")
        self._has_references = True


class SharingManager(Answer):
    """Answer implementation which automatically shares all returned references with the user."""

    class _LockManager:
        _storage: dict[str, asyncio.Lock] = {}
        _storage_lock = asyncio.Lock()

        async def get(self, key: str) -> asyncio.Lock:
            async with self._storage_lock:
                if key not in self._storage:
                    self._storage[key] = asyncio.Lock()
                return self._storage[key]

    @inject
    def __init__(self, wrapped_answer: Answer, *, file_storage: FileStorage = NotImplemented):
        self._wrapped_answer = wrapped_answer
        self._file_storage = file_storage
        self._urls: dict[str, str] = {}
        self._lock_manager = self._LockManager()

    def __enter__(self) -> Self:
        self._wrapped_answer.__enter__()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, tb: TracebackType | None
    ):
        self._wrapped_answer.__exit__(exc_type, exc_value, tb)

    @inject
    async def share_document(
        self, doc: RetrievedDocument, *, document_service: DocumentService = NotImplemented
    ) -> RetrievedDocument:
        async with await self._lock_manager.get(doc.source_url):
            if doc.source_url not in self._urls:
                document = await document_service.get_document(doc.source_id)
                self._urls[doc.source_url] = await document.share_with_user()

        assert doc.source_url in self._urls

        return doc.model_copy(update={"source_url": self._urls[doc.source_url]})

    def append_content(self, content: str):
        self._wrapped_answer.append_content(content)

    async def add_citation(self, doc: RetrievedDocument):
        await self._wrapped_answer.add_citation(await self.share_document(doc))

    def create_stage(self, name: str, *, debug: bool = False, timed: bool = True) -> AnswerStage:
        return self._wrapped_answer.create_stage(name, debug=debug, timed=timed)


class DialStage(AnswerStage):
    _start: float | None = None
    _ping_task: asyncio.Task | None = None

    def __init__(self, stage: Stage, *, timed: bool = True, show_debug_info: bool = True):
        self._stage = stage
        self._timed = timed
        self._show_debug_info = show_debug_info

    def __enter__(self) -> Self:
        self._stage.__enter__()
        if self._timed:
            self._start = time.perf_counter()
            self._ping_task = asyncio.create_task(self._periodic_ping())
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, tb: TracebackType | None
    ):
        if self._start is not None:
            with suppress(Exception):
                end = time.perf_counter()
                self._stage.append_name(f" [{end - self._start:.2f}s]")
                self._start = None

        if exc_value:
            with suppress(Exception):
                logger.warning(str(exc_value), exc_info=exc_value)
                self._add_exception(exc_type, exc_value, tb)

        if self._ping_task:
            with suppress(Exception):
                self._ping_task.cancel()
                self._ping_task = None

        return self._stage.__exit__(exc_type, exc_value, tb)

    def append_content(self, content: str):
        self._stage.append_content(content)

    async def add_citation(self, doc: RetrievedDocument): ...

    def _add_exception(
        self, exc_type: type[BaseException] | None, exc_value: BaseException, tb: TracebackType | None
    ):
        trace_id = "unknown"
        span_id = "unknown"

        if ((span := get_current_span()) != INVALID_SPAN) and (
            (ctx := span.get_span_context()) != INVALID_SPAN_CONTEXT
        ):
            trace_id = f"{ctx.trace_id:032x}"
            span_id = f"{ctx.span_id:016x}"

        self._stage.append_content("Execution completed with error.\n\n")
        self._stage.append_content(f"```\n{str(exc_value)}\n\n{trace_id=}\n{span_id=}\n```\n\n")

        if self._show_debug_info:
            stack_trace = "".join(traceback.format_exception(exc_type, exc_value, tb))
            self._stage.append_content(f"```python\n{stack_trace}\n```\n\n")

    async def _periodic_ping(self):
        while True:
            try:
                await asyncio.sleep(15)
            except asyncio.CancelledError:
                break
            self._stage.content_stream.write("")


class DialAnswer(Answer):
    def __init__(self, choice: Choice, settings: ChatSettings):
        self._choice = choice
        self._settings = settings

        self._used_references: list[tuple[int, int]] = []

        self._last_citation_id: uuid.UUID | None = None
        self._annotations_counter = itertools.count()

    def __enter__(self) -> Self:
        self._choice.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        tb: TracebackType | None,
    ):
        return self._choice.__exit__(exc_type, exc_value, tb)

    def append_content(self, content: str):
        self._choice.append_content(content)
        if self._last_citation_id and content.strip():
            self._last_citation_id = None

    async def add_citation(self, doc: RetrievedDocument):
        if self._settings.enable_annotations:
            self._cite_as_annotation(doc)
        else:
            self._cite_as_attachment(doc)

    def _cite_as_annotation(self, doc: RetrievedDocument):
        if self._last_citation_id is None:
            self._last_citation_id = uuid.uuid4()
            assert self._last_citation_id
            self._choice.append_content(
                f'<{CITATION_TAG} data-id="{self._last_citation_id.hex}"></{CITATION_TAG}>'
            )

        self._choice.send_chunk(
            ArbitraryChunk({
                "choices": [
                    {
                        "index": self._choice.index,
                        "finish_reason": None,
                        "delta": {
                            "custom_content": {
                                "annotations": [
                                    _create_annotation(
                                        doc, next(self._annotations_counter), self._last_citation_id
                                    ),
                                ]
                            },
                        },
                    }
                ]
            })
        )

    def _cite_as_attachment(self, doc: RetrievedDocument):
        reference_key = (doc.source_id, doc.source_page_number)

        if reference_key not in self._used_references:
            self._used_references.append(reference_key)
            reference_index = len(self._used_references)

            self._choice.add_attachment(
                _create_attachment(
                    doc,
                    reference_index,
                    thumbnails=self._settings.enable_thumbnails,
                )
            )
        else:
            reference_index = self._used_references.index(reference_key) + 1

        self._choice.append_content(f" [{reference_index}]")

    def create_stage(self, name: str, *, debug: bool = False, timed: bool = True) -> AnswerStage:
        if debug:
            if not self._settings.enable_debug_stages:
                return NoopStage()
            name = f"[DEBUG] {name}"

        return DialStage(
            self._choice.create_stage(name),
            timed=timed,
            show_debug_info=self._settings.enable_debug_stages,
        )


def _create_annotation(doc: RetrievedDocument, index: int, citation_id: uuid.UUID) -> dict[str, Any]:
    selector = (
        {
            "type": "pdf_bbox",
            "page": doc.source_page_number,
            "x1": 0,
            "y1": 0,
            "x2": 0,
            "y2": 0,
        }
        if doc.source_mime_type == "application/pdf" and doc.source_page_number
        else None
    )

    return remove_nones({
        "index": index,
        "target": {
            "selector": {
                "type": "html_tag",
                "tag": CITATION_TAG,
                "id": citation_id.hex,
            },
        },
        "body": {
            "title": doc.source_title + f", page {doc.source_page_number}" if doc.source_page_number else "",
            "quote": _create_document_quote(doc, False),
            "source": {
                "type": "attachment",
                "attachment": {
                    "type": doc.source_mime_type,
                    "title": doc.source_name,
                    "url": doc.source_url,
                },
            },
            "selector": selector,
        },
    })


def _create_document_quote(doc: RetrievedDocument, thumbnails) -> str:
    result = ""

    for chunk in doc.chunks:
        if isinstance(chunk, TextChunk):
            result += f"{chunk.text}\n\n"

        elif isinstance(chunk, ImageChunk):
            image_title = f"Image of {chunk.image_type}"
            if doc.source_page_number:
                image_title += (
                    f" #{doc.source_page_number}"
                    if chunk.image_type == ImageType.page
                    else f", page #{doc.source_page_number}"
                )
            if thumbnails:
                image_url = _create_thumbnail(chunk)
                result += f'![{image_title}]({image_url} "{image_title}")\n\n'
            else:
                result += f"[{image_title}]\n\n"

    return result.rstrip()


def _create_attachment(doc: RetrievedDocument, citation_index: int, *, thumbnails: bool = False):
    title = f"[{citation_index}] {doc.source_title}"
    if doc.source_page_number:
        title += f", page {doc.source_page_number}"

    return Attachment(
        type="text/markdown",
        title=title,
        data=_create_document_quote(doc, thumbnails) or " ",
        reference_url=(
            f"{doc.source_url}#page={doc.source_page_number}" if doc.source_page_number else doc.source_url
        ),
    )


def _create_thumbnail(chunk: ImageChunk, size: int = 256) -> str:
    """
    Create thumbnail for given image chunk.

    :param chunk: the chunk with original image
    :param size: requested thumbnail size in pixels
    :return: base64-encoded data url for created thumbnail
    """
    chunk_image = Image.open(io.BytesIO(chunk.content))
    chunk_image.thumbnail(size=(size, size), resample=Resampling.BICUBIC)
    with io.BytesIO() as fp:
        chunk_image.save(fp, format="jpeg")
        return DataURI.make(
            mimetype="image/jpeg",
            charset=None,
            base64=True,
            data=fp.getvalue(),
        )
