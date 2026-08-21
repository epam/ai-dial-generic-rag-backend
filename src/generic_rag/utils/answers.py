import asyncio
import io
import logging
import time
import traceback
from contextlib import suppress
from types import TracebackType
from typing import Self

from aidial_sdk.chat_completion import Attachment, Choice, Stage
from datauri import DataURI
from injection import inject
from opentelemetry.trace import INVALID_SPAN, INVALID_SPAN_CONTEXT, get_current_span
from PIL import Image
from PIL.Image import Resampling

from generic_rag.types import (
    Answer,
    AnswerStage,
    FileStorage,
    ImageChunk,
    RetrievedDocument,
    TextChunk,
)

logger = logging.getLogger(__name__)


class NoopStage(AnswerStage):
    """Stage that does nothing."""

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, tb: TracebackType | None
    ): ...

    def append_content(self, content: str): ...

    async def add_citation(self, citation_index: int, doc: RetrievedDocument): ...

    async def add_reference(self, citation_index: int, doc: RetrievedDocument): ...


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

    async def add_citation(self, citation_index: int, doc: RetrievedDocument):
        self.append_content(f"[{doc.source_id, doc.source_page_number}]")
        self._has_references = True

    async def add_reference(self, citation_index: int, doc: RetrievedDocument):
        self._has_references = True
        # do nothing for now


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

    class _StageWrapper(AnswerStage):
        def __init__(self, wrapped_stage: AnswerStage, sharing_manager: "SharingManager"):
            self._wrapped_stage = wrapped_stage
            self._sharing_manager = sharing_manager

        def __enter__(self) -> Self:
            self._wrapped_stage.__enter__()
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            tb: TracebackType | None,
        ):
            self._wrapped_stage.__exit__(exc_type, exc_value, tb)

        def append_content(self, content: str):
            self._wrapped_stage.append_content(content)

        async def add_citation(self, citation_index: int, doc: RetrievedDocument):
            await self._wrapped_stage.add_citation(citation_index, doc)

        async def add_reference(self, citation_index: int, doc: RetrievedDocument):
            await self._wrapped_stage.add_reference(
                citation_index, await self._sharing_manager.share_document(doc)
            )

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

    async def share_document(self, doc: RetrievedDocument) -> RetrievedDocument:
        async with await self._lock_manager.get(doc.source_url):
            if doc.source_url not in self._urls:
                self._urls[doc.source_url] = await self._file_storage.copy_file_to_user(
                    doc.source_url,
                    doc.source_display_name,
                )

        assert doc.source_url in self._urls

        return doc.model_copy(update={"source_url": self._urls[doc.source_url]})

    def append_content(self, content: str):
        self._wrapped_answer.append_content(content)

    async def add_citation(self, citation_index: int, doc: RetrievedDocument):
        await self._wrapped_answer.add_citation(citation_index, doc)

    async def add_reference(self, citation_index: int, doc: RetrievedDocument):
        await self._wrapped_answer.add_reference(citation_index, await self.share_document(doc))

    def create_stage(self, name: str, *, debug: bool = False, timed: bool = True) -> AnswerStage:
        stage = self._wrapped_answer.create_stage(name, debug=debug, timed=timed)
        if not isinstance(stage, NoopStage):
            return self._StageWrapper(stage, self)
        return stage


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

    async def add_citation(self, citation_index: int, doc: RetrievedDocument): ...

    async def add_reference(self, citation_index: int, doc: RetrievedDocument):
        self._stage.add_attachment(create_attachment(doc, citation_index))

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
    def __init__(self, choice: Choice, *, enable_debug_stages: bool = True):
        self._choice = choice
        self._enable_debug_stages = enable_debug_stages

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

    async def add_citation(self, citation_index: int, doc: RetrievedDocument):
        self._choice.append_content(f" [{citation_index}]")

    async def add_reference(self, citation_index: int, doc: RetrievedDocument):
        self._choice.add_attachment(create_attachment(doc, citation_index))

    def create_stage(self, name: str, *, debug: bool = False, timed: bool = True) -> AnswerStage:
        if debug:
            if not self._enable_debug_stages:
                return NoopStage()
            name = f"[DEBUG] {name}"

        return DialStage(
            self._choice.create_stage(name),
            timed=timed,
            show_debug_info=self._enable_debug_stages,
        )


def create_attachment(doc: RetrievedDocument, citation_index: int):
    data = ""

    for chunk in doc.chunks:
        if isinstance(chunk, TextChunk):
            data += f"{chunk.text}\n\n"

        elif isinstance(chunk, ImageChunk):
            image_title = f"Image of page #{chunk.page_number}"
            image_url = create_thumbnail(chunk)
            data += f'![{image_title}]({image_url} "{image_title}")\n\n'

    return Attachment(
        type="text/markdown",
        title=f"[{citation_index}] {doc.source_display_name}",
        data=data.rstrip() or " ",
        reference_url=(
            f"{doc.source_url}#page={doc.source_page_number}" if doc.source_page_number else doc.source_url
        ),
    )


def create_thumbnail(chunk: ImageChunk, size: int = 256) -> str:
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
