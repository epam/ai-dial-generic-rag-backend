import csv
import io
import json
import logging
import traceback
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from contextvars import ContextVar, Token

from aidial_sdk.chat_completion import ChatCompletion, Choice, Request, Response, Stage
from aidial_sdk.deployment.configuration import ConfigurationRequest, ConfigurationResponse
from aidial_sdk.exceptions import InternalServerError
from langchain_community.callbacks import get_openai_callback
from langchain_core.documents import Document as LangchainDocument
from opentelemetry.trace import INVALID_SPAN, INVALID_SPAN_CONTEXT, get_current_span
from pydantic import SecretStr, ValidationError
from pydantic_partial import create_partial_model

from generic_rag.channel import Channel, RequestConfig
from generic_rag.scope import ChannelBindings
from generic_rag.services.chunk_sources_manager import ChunkSource
from generic_rag.types import AnswerCallback, AnswerGenerator, AnyChunk, RetrievalStageListener, Retriever
from generic_rag.utils.attachment import create_attachment
from generic_rag.utils.profile import timed_stage

_current_stage: ContextVar[Stage] = ContextVar("_current_stage")

logger = logging.getLogger(__name__)


class ChannelCompletion(ChatCompletion):
    _enable_debug_output: bool = True

    async def chat_completion(self, request: Request, response: Response) -> None:
        """ Chat completion entrypoint. """
        async with ChannelBindings(SecretStr(request.api_key), request.dial_application_id).scope.adefine():
            with response.create_single_choice() as choice:
                try:
                    with timed_stage(choice, "[DEBUG] channel configuration") as stage:
                        channel = await Channel.get_current_channel()

                        stage.append_content(f"application id: `{request.dial_application_id}`\n\n")
                        stage.append_content(
                            f"```json\n{json.dumps(channel.dump_config(), indent=2)}\n```\n\n"
                        )

                    await self._channel_completion(choice, request, response, channel)

                except Exception as e:
                    with choice.create_stage("[DEBUG] internal error") as stage:
                        await report_exception_to_stage(stage, e)
                        raise InternalServerError(
                            f"Unexpected error: {str(e)}"
                        ) from e

    async def _channel_completion(self, choice: Choice, request: Request, response: Response, channel: Channel):
        """ Chat completion logic for specific channel. """
        with timed_stage(choice, "[DEBUG] request configuration") as stage:
            request_config_model = await RequestConfig.get_dynamic_model()
            raw_request = request.custom_fields.configuration if request.custom_fields else None
            try:
                request_config = request_config_model.create(
                    defaults=channel.request_config,
                    overrides=raw_request,
                )
            except ValidationError as e:
                stage.append_content(f"```\n{e.__class__.__name__}: {str(e)}\n```\n\n")
                if isinstance(raw_request, dict):
                    stage.append_content(f"```json\n{json.dumps(raw_request, indent=2)}\n```\n\n")
                raise InternalServerError(
                    "Unable to process request configuration"
                ) from e
            else:
                stage.append_content(
                    f"```json\n{request_config.model_dump_json(indent=2, exclude_none=True)}\n```\n\n"
                )

        last_message = str(request.messages[-1].content)

        retriever = Retriever.create(
            request_config.retriever
        ).use_listener(
            ChatCompletionRetrievalStageListener(choice, self._enable_debug_output)
        )

        answer_generator = AnswerGenerator.create(
            request_config.generation
        )

        with get_openai_callback() as cb:
            await answer_generator.invoke(last_message, retriever, ChatCompletionAnswerCallback(choice))

            with choice.create_stage("[DEBUG] token usage") as stage:
                stage.append_content(f"```text\n{str(cb)}\n```\n\n")

            response.set_usage(cb.prompt_tokens, cb.completion_tokens)

    async def configuration(self, request: ConfigurationRequest) -> ConfigurationResponse | dict:
        """ Chat completion configuration entrypoint. """
        async with ChannelBindings(SecretStr(request.api_key), request.dial_application_id).scope.adefine():
            configuration_model = create_partial_model(
                await RequestConfig.get_dynamic_model()
            )
            return configuration_model.model_json_schema()


class ChatCompletionAnswerCallback(AnswerCallback):
    def __init__(self, choice: Choice):
        self._choice = choice
        self._used_references: list[int] = []

    def append_content(self, content: str):
        self._choice.append_content(content)

    def append_reference(self, reference_index: int, retrieved_doc: LangchainDocument):
        chunks = retrieved_doc.metadata.get("chunks", [])

        if reference_index not in self._used_references:
            self._used_references.append(reference_index)
            cite_index = len(self._used_references)
            if attachment := create_attachment(chunks, cite_index=cite_index):
                self._choice.add_attachment(attachment)
        else:
            cite_index = self._used_references.index(reference_index) + 1

        self.append_content(f" [{cite_index}]")


class ChatCompletionRetrievalStageListener(RetrievalStageListener):
    """ Listener that mirrors retrieval stages as stages of chat completion Choice. """

    def __init__(self, choice: Choice, enable_debug_output: bool):
        self._choice = choice
        self._enable_debug_output = enable_debug_output

    @property
    def _stage(self) -> Stage:
        return _current_stage.get()

    def begin(self, stage_name: str) -> AbstractAsyncContextManager["RetrievalStageListener"]:
        """ Return context manager that wraps single stage of retrieval process. """

        @asynccontextmanager
        async def _context_manager():
            token = None
            try:
                with timed_stage(self._choice, stage_name) as stage:
                    token = _current_stage.set(stage)
                    yield self
            finally:
                if isinstance(token, Token):
                    _current_stage.reset(token)

        return _context_manager()

    def log_message(self, message: str):
        """ Called to report a message related to stage execution. """
        self._stage.append_content(f"{message}\n\n")

    async def on_retrieval_result(self, retrieved_docs: list[LangchainDocument]):
        """ Called when receive retrieval results. """
        self._stage.append_content(f"Found {len(retrieved_docs)} reference(s)\n\n")

        if chunk_summary := self._get_chunk_summary(retrieved_docs):
            with io.StringIO() as fp:
                writer = csv.DictWriter(fp, fieldnames=chunk_summary[0].keys())
                writer.writeheader()
                writer.writerows(chunk_summary)
                self._stage.append_content(f"```csv\n{fp.getvalue()}\n```\n")

        for i, document in enumerate(retrieved_docs, start=1):
            chunks: list[AnyChunk] = document.metadata.get("chunks", [])

            if attachment := create_attachment(chunks, cite_index=i):
                self._stage.add_attachment(attachment)

    async def on_error(self, e: Exception):
        """ Called in case of errors. """
        await report_exception_to_stage(self._stage, e)

    @staticmethod
    def _get_chunk_summary(retrieved_docs: list[LangchainDocument]):
        result = []

        for i, document in enumerate(retrieved_docs, start=1):
            chunks: list[AnyChunk] = document.metadata.get("chunks", [])
            extras = {k: v for k, v in document.metadata.items() if k not in ("identity", "chunks")}

            result.extend(
                {
                    "#": f"[{i}]",
                    "source_name": ChunkSource.from_chunk(chunk).source_display_name,
                    "page_number": chunk.page_number or "?",
                }
                | extras
                | chunk.get_identity().model_dump()
                for chunk in chunks
            )

        return result


async def report_exception_to_stage(stage: Stage, exception: Exception, enable_stack_trace: bool = True):
    trace_id = "unknown"
    span_id = "unknown"

    if (
        ((span := get_current_span()) != INVALID_SPAN) and
        ((ctx := span.get_span_context()) != INVALID_SPAN_CONTEXT)
    ):
        trace_id = f"{ctx.trace_id:032x}"
        span_id = f"{ctx.span_id:016x}"

    logger.warning(str(exception), exc_info=exception if enable_stack_trace else None)

    stage.append_content(f"Internal error (`{trace_id=}`, `{span_id=}`)\n\n")
    stage.append_content(f"```\n{str(exception)}\n```\n\n")

    if enable_stack_trace:
        stack_trace = "".join(traceback.format_exception(exception))
        stage.append_content(f"```\n{stack_trace}\n```\n\n")
