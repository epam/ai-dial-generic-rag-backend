import base64
import enum
import os
from collections.abc import Sequence
from contextlib import AsyncExitStack
from enum import StrEnum
from typing import Annotated, Any, Literal, NamedTuple, cast

from annotated_types import Gt
from fastapi import FastAPI
from fastmcp import FastMCP
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.server.providers import LocalProvider
from fastmcp.server.transforms import Transform
from fastmcp.tools import Tool
from fastmcp.tools.tool_transform import ArgTransform, TransformedTool
from injection import asfunction, inject
from langchain_core.documents import Document as LangchainDocument
from mcp import types as mt
from mcp.types import ImageContent, TextContent, ToolAnnotations
from pydantic import BaseModel, Field, SecretStr, TypeAdapter, create_model

from generic_rag.channel import Channel, RequestConfig
from generic_rag.components.retrieval.document_selector import (
    ExactDocumentsDocumentSelector,
    ExplicitDocumentSelector,
)
from generic_rag.scope import ChannelBindings
from generic_rag.services.chunk_service import ChunkService
from generic_rag.services.document_matcher import DocumentMatcher, DocumentMatcherConfig, SingleFilterModel
from generic_rag.services.document_service import DocumentService
from generic_rag.services.document_stats_service import DocumentStats, DocumentStatsService
from generic_rag.services.metadata_service import MetadataService
from generic_rag.types import (
    AnswerCallback,
    AnswerGenerator,
    AnyChunk,
    ChunkSource,
    ChunkType,
    Document,
    ImageChunk,
    ImageType,
    Retriever,
    TextChunk,
)
from generic_rag.utils.pagination import PaginatedResults, Pagination

GET_PAGES_LIMIT = TypeAdapter(Annotated[int, Gt(0)]).validate_python(
    os.getenv("MCP_GET_PAGES_LIMIT", "10"),
)

provider = LocalProvider()


@enum.unique
class ToolName(StrEnum):
    LIST_DOCUMENTS = "list_documents_unordered"
    GET_PAGES = "get_pages"
    RETRIEVE_TEXT_CHUNKS = "retrieve_text_chunks"
    RAG_SEARCH = "rag_search"


class DocumentMetadata(BaseModel):
    """Document summary."""

    id: Annotated[int, Field(description="Unique id of this document")]
    title: Annotated[str, Field(description="Document title")]
    number_of_pages: Annotated[int, Field(description="Number of document pages", ge=0)]

    @classmethod
    @inject
    async def get_dynamic_model[T: DocumentMetadata](
        cls: type[T], metadata_service: MetadataService = NotImplemented
    ) -> type["DocumentMetadata"]:
        if filterable_fields := metadata_service.get_filterable_fields():
            # noinspection bad-return
            return create_model(
                cls.__name__,
                __base__=cls,
                __doc__=cls.__doc__,
                **{
                    field_name: (
                        field_info.rebuild_annotation(),
                        field_info,
                    )
                    for field_name, field_info in filterable_fields
                },
            )
        return cls

    @classmethod
    def create(cls, document: Document, stats: DocumentStats | None):
        return cls.model_validate(
            dict(
                id=document.id,
                title=document.display_name,
                number_of_pages=stats and stats.number_of_pages or 0,
                **document.metadata,
            )
        )


@provider.tool(
    name=ToolName.LIST_DOCUMENTS, annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False)
)
@asfunction()
class ListDocumentsTool(NamedTuple):
    channel: Channel
    document_service: DocumentService
    stats_service: DocumentStatsService

    async def __call__(
        self,
        metadata_filter: Annotated[
            dict[str, Any] | None,  # NOTE: this type is going to be overwritten by ArgTransform
            Field(description="Filter by metadata fields"),
        ] = None,
        offset: Annotated[int, Field(ge=0, description="Offset, for pagination")] = 0,
        limit: Annotated[
            int, Field(ge=0, description="Maximum number of results to return, for pagination")
        ] = 25,
    ) -> dict[str, Any]:
        """
        List indexed documents along with their metadata.
        Allows to filter by metadata fields and paginate the results.
        Results are unsorted and unordered. Pagination does not imply ranking or recency.
        Do not use the first page of results to infer “latest”, “top”, “first”, or “last” documents.
        """
        if metadata_filter:
            matcher_config_model = await DocumentMatcherConfig.get_dynamic_model()
            document_matcher = DocumentMatcher(
                self.channel.channel_key, matcher_config_model.model_validate({"filters": [metadata_filter]})
            )
        else:
            document_matcher = None

        pagination = Pagination(offset, limit)
        documents_list = await self.document_service.list_documents(pagination, document_matcher)
        documents_stats: dict[int, DocumentStats] = {
            doc_stats.document_id: doc_stats
            for doc_stats in await self.stats_service.get_document_stats(*[
                doc.id for doc in documents_list.results
            ])
        }

        document_metadata_model = await DocumentMetadata.get_dynamic_model()

        result = PaginatedResults.create(
            results=[
                document_metadata_model.create(document, documents_stats.get(document.id))
                for document in documents_list.results
            ],
            pagination=pagination,
            total_count=documents_list.total_count,
        )
        return result.model_dump(exclude_unset=True)


@provider.tool(name=ToolName.GET_PAGES, annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
@asfunction
class GetPagesTool(NamedTuple):
    chunk_service: ChunkService

    async def __call__(
        self,
        document_id: Annotated[int, Field(description="ID of the document", ge=1)],
        page_start: Annotated[int, Field(description="Start page of the document (1 based)", ge=1)],
        page_end: Annotated[int, Field(description="End page of the document (1 based)", ge=1)],
        retrieve_type: Annotated[
            Literal["text", "image", "both"], Field(alias="type", description="Content type to retrieve")
        ] = "both",
    ) -> list[TextContent | ImageContent]:
        """Returns the full content of specific document's page range (text, image, or both)."""
        if page_start > page_end:
            raise ValueError("'page_start' cannot be greater than 'page_end'")
        if page_end - page_start > GET_PAGES_LIMIT:
            raise ValueError(
                f"you can request maximum {GET_PAGES_LIMIT} pages in single tool call "
                f"({page_end - page_start} pages requested)"
            )

        doc_pages = [(document_id, page_idx) for page_idx in range(page_start, page_end + 1)]

        match retrieve_type:
            case "text":
                chunks = await self.chunk_service.get_chunks_by_pages(*doc_pages, chunk_type=ChunkType.text)
            case "image":
                chunks = await self.chunk_service.get_chunks_by_pages(*doc_pages, chunk_type=ChunkType.image)
            case "both":
                chunks = await self.chunk_service.get_chunks_by_pages(*doc_pages)

        text_content: dict[int, TextContent] = {}
        image_content: dict[int, ImageContent] = {}

        for chunk in chunks:
            if isinstance(chunk, TextChunk):
                assert chunk.page_number is not None
                if content_block := text_content.get(chunk.page_number):
                    content_block.text += "\n" + chunk.text
                else:
                    text_content[chunk.page_number] = TextContent(type="text", text=chunk.text)

            elif isinstance(chunk, ImageChunk) and chunk.image_type == ImageType.page:
                assert chunk.page_number is not None
                image_content[chunk.page_number] = ImageContent(
                    type="image", data=base64.b64encode(chunk.content).decode(), mimeType=chunk.mime_type
                )

        return list(self._get_result(document_id, text_content, image_content))

    @staticmethod
    def _get_result(
        document_id: int, page_text: dict[int, TextContent], page_images: dict[int, ImageContent]
    ):
        for page_idx in sorted(set(page_text.keys()) | set(page_images.keys())):
            yield TextContent(type="text", text=f"[Document {document_id}, Page {page_idx}]")
            if content_block := page_text.get(page_idx):
                yield content_block
            if content_block := page_images.get(page_idx):
                yield content_block


class RetrievedChunk(BaseModel):
    document_id: Annotated[int, Field(description="`id` of related document")]
    chunk_id: Annotated[int, Field(description="`id` of chunk within the document")]
    text: Annotated[str, Field(description="text of retrieved chunk")]
    page_number: Annotated[int | None, Field(description="number of page where this chunk was extracted")] = (
        None
    )
    metadata: Annotated[dict | None, Field(description="metadata of related document", default_factory=dict)]

    @classmethod
    def create[T: RetrievedChunk](
        cls: type[T],
        chunks: list[AnyChunk],
        metadata_field_names: set[str] | None = None,
    ) -> T | None:
        for chunk in chunks:
            if isinstance(chunk, TextChunk):
                chunk_source = ChunkSource.from_chunk(chunk)
                metadata = chunk_source.source_metadata
                if metadata_field_names is not None:
                    metadata = {k: v for k, v in metadata.items() if k in metadata_field_names}
                return cls(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    page_number=chunk.page_number,
                    metadata=metadata,
                )
            # NOTE: don't include image chunks -
            # they bloat the context and likely don't bring much value.
            # we expose get_page tool with image mode.
            # we can re-enable image chunks retrieval later if needed.
            # if (
            #     isinstance(chunk, ImageChunk) and
            #     chunk.image_type == ImageType.page and
            #     result is not None
            # ):
            #     result.page_image = chunk.content
        return None


def _get_retriever_overrides(document_ids: list[int] | None, metadata_filter: dict[str, Any] | None):
    if document_ids:
        return {
            "document_selector": {
                "type": ExactDocumentsDocumentSelector.get_qualifier(),
                "document_ids": document_ids,
            }
        }
    if metadata_filter:
        return {
            "document_selector": {
                "type": ExplicitDocumentSelector.get_qualifier(),
                "filters": [metadata_filter],
            }
        }
    return {}


@provider.tool(
    name=ToolName.RETRIEVE_TEXT_CHUNKS, annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False)
)
@asfunction()
class DataRetrievalTool(NamedTuple):
    channel: Channel
    metadata_service: MetadataService

    async def __call__(
        self,
        query: Annotated[str, Field(description="The search query")],
        document_ids: Annotated[
            list[int] | None, Field(description="Restrict search to specific documents")
        ] = None,
        metadata_filter: Annotated[
            dict[str, Any] | None,  # NOTE: this type is going to be overwritten by ArgTransform
            Field(description="Filter by metadata fields. Ignored if document_ids is provided"),
        ] = None,
    ) -> list[RetrievedChunk]:
        """
        Run retrieval part of RAG search pipeline.
        Returns raw chunks relevant to a given query.
        Allows to restrict retrieval to specific documents or to filter by metadata fields.
        """
        request_config_model = await RequestConfig.get_dynamic_model()
        request_config = request_config_model.create(
            defaults=self.channel.request_config,
            overrides={
                "retriever": _get_retriever_overrides(document_ids, metadata_filter),
            },
        )

        retriever = Retriever.create(request_config.retriever)
        metadata_field_names = self.metadata_service.get_mcp_retrieve_chunks_field_names()

        return [
            retrieved_chunk
            for doc in await retriever.invoke(query)
            if (
                retrieved_chunk := RetrievedChunk.create(
                    doc.metadata.get("chunks", []),
                    metadata_field_names=metadata_field_names,
                )
            )
            is not None
        ]


class SearchToolAnswerCallback(AnswerCallback):
    def __init__(self):
        self._content: str = ""
        self._has_references = False

    @property
    def content(self):
        return self._content

    @property
    def has_references(self) -> bool:
        return self._has_references

    def append_content(self, content: str):
        self._content += content

    def append_reference(self, reference_index: int, retrieved_doc: LangchainDocument):
        if chunks := cast(list[AnyChunk], retrieved_doc.metadata.get("chunks", [])):
            chunk = chunks[0]
            self.append_content(f"[{chunk.document_id, chunk.page_number}]")
            self._has_references = True


_SEARCH_QUERY_DESCRIPTION = """\
Natural-language search query — a question phrased the way one person would ask another.

Each query should target a single piece of information.
For compound requests (e.g. asking for a number and a reason), issue multiple calls — one atomic question per call.

Do not include filtering or scoping instructions in the query; use other arguments for that.
"""


@provider.tool(
    name=ToolName.RAG_SEARCH, annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False)
)
@asfunction()
class SearchTool(NamedTuple):
    channel: Channel

    async def __call__(
        self,
        query: Annotated[str, Field(description=_SEARCH_QUERY_DESCRIPTION)],
        document_ids: Annotated[
            list[int] | None, Field(description="Restrict search to specific documents")
        ] = None,
        metadata_filter: Annotated[
            dict[str, Any] | None,  # NOTE: this type is going to be overwritten by ArgTransform
            Field(description="Filter by metadata fields. Ignored if document_ids is provided"),
        ] = None,
    ) -> TextContent:
        """
        Run RAG search pipeline (retrieval + generation) across indexed documents.
        Returns LLM-generated summary (not raw retrieval artifacts).
        Allows to restrict search to specific documents or to filter by metadata fields.
        Response contains document citations in `(document, page)` format.
        """
        request_config_model = await RequestConfig.get_dynamic_model()
        request_config = request_config_model.create(
            defaults=self.channel.request_config,
            overrides={
                "retriever": _get_retriever_overrides(document_ids, metadata_filter),
                "generation": {
                    "type": "default",
                },
            },
        )

        retriever = Retriever.create(request_config.retriever)
        answer_generator = AnswerGenerator.create(request_config.generation)
        callback = SearchToolAnswerCallback()

        await answer_generator.invoke(query, retriever, callback)

        text = callback.content
        if not callback.has_references:
            text += "\n\nNo references found"
        return TextContent(type="text", text=text)


class DynamicSchemasTransform(Transform):
    """Transform that adds correct dynamic schemas."""

    async def list_tools(self, tools: Sequence[Tool]) -> Sequence[Tool]:
        """List tools with transformation applied."""
        single_filter_model = await SingleFilterModel.get_dynamic_model()
        document_metadata_model = await DocumentMetadata.get_dynamic_model()

        result: list[Tool] = []

        for current_tool in await super().list_tools(tools):
            transformed = current_tool

            if current_tool.name in {
                ToolName.LIST_DOCUMENTS,
                ToolName.RETRIEVE_TEXT_CHUNKS,
                ToolName.RAG_SEARCH,
            }:
                transformed = TransformedTool.from_tool(
                    tool=transformed,
                    transform_args={
                        # NOTE: it's important to match schema here to the original one.
                        # specifically, we need to match `x | None` signature.
                        # if original is `dict | None`,
                        # setting `single_filter_model` here (not `single_filter_model | None`)
                        # leads to mistakes in json schema and GPT models failing to use the tool.
                        "metadata_filter": ArgTransform(type=single_filter_model | None)
                    },
                )

            if current_tool.name == ToolName.LIST_DOCUMENTS:
                # noinspection PyTypeHints
                transformed = TransformedTool.from_tool(
                    tool=transformed,
                    output_schema=TypeAdapter(PaginatedResults[document_metadata_model]).json_schema(),
                )

            result.append(transformed)

        return result


provider.add_transform(DynamicSchemasTransform())


class ChannelMiddleware(Middleware):
    """Middleware that performs initialization Channel initialization for requests."""

    async def on_request(
        self, context: MiddlewareContext[mt.Request[Any, Any]], call_next: CallNext[mt.Request[Any, Any], Any]
    ) -> Any:
        if (
            context.fastmcp_context
            and context.fastmcp_context.request_context
            and (request := context.fastmcp_context.request_context.request)
        ):
            api_key = request.headers.get("api-key")
            application_id = request.headers.get("x-dial-application-id")

            async with ChannelBindings(SecretStr(api_key), application_id).scope.adefine():
                return await super().on_request(context, call_next)

        return await super().on_request(context, call_next)


async def setup_mcp(app: FastAPI, exit_stack: AsyncExitStack):
    mcp_app = FastMCP(
        name="Generic RAG MCP",
        instructions="This server provides tools to interact with generic-rag",
        providers=[provider],
        middleware=[ChannelMiddleware()],
    )
    await exit_stack.enter_async_context(mcp_app.lifespan())

    http_app = mcp_app.http_app(path="/streamable-http", stateless_http=True)
    app.mount("/mcp", http_app)
    await exit_stack.enter_async_context(http_app.lifespan(http_app))
