import base64
import itertools
import os
from asyncio import TaskGroup
from collections.abc import Iterable, Sequence
from contextlib import AsyncExitStack
from typing import Annotated, Any, Literal, Self

import annotated_types
from annotated_types import Gt
from fastapi import FastAPI
from fastmcp import FastMCP
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.server.providers import LocalProvider
from fastmcp.server.transforms import Transform
from fastmcp.tools import Tool
from fastmcp.tools.tool_transform import ArgTransform, TransformedTool
from injection import afind_instance, inject
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
from generic_rag.services.document_matcher import DocumentMatcherConfig, SingleFilterModel
from generic_rag.services.document_service import DocumentService
from generic_rag.services.document_stats_service import DocumentStats, DocumentStatsService
from generic_rag.services.metadata_service import MetadataService
from generic_rag.types import (
    AnswerGenerator,
    ChunkType,
    Document,
    FileStorage,
    ImageChunk,
    ImageType,
    RetrievedDocument,
    Retriever,
    TextChunk,
)
from generic_rag.utils.answers import PlainAnswer
from generic_rag.utils.pagination import PaginatedResults, Pagination

GET_PAGES_LIMIT = TypeAdapter(Annotated[int, Gt(0)]).validate_python(
    os.getenv("MCP_GET_PAGES_LIMIT", "10"),
)

provider = LocalProvider()


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


@provider.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
async def list_documents_unordered(
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
    document_service = await afind_instance(DocumentService)
    stats_service = await afind_instance(DocumentStatsService)

    matcher_config = (
        (await DocumentMatcherConfig.get_dynamic_model()).model_validate(
            {"filters": [metadata_filter]},
        )
        if metadata_filter
        else None
    )

    pagination = Pagination(offset, limit)
    documents_list = await document_service.list_documents(pagination, matcher_config)
    documents_stats: dict[int, DocumentStats] = {
        doc_stats.document_id: doc_stats
        for doc_stats in await stats_service.get_document_stats(*[doc.id for doc in documents_list.results])
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


@provider.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
async def get_pages(
    document_id: Annotated[int, Field(description="ID of the document", ge=1)],
    page_start: Annotated[int, Field(description="Start page of the document (1 based)", ge=1)],
    page_end: Annotated[int, Field(description="End page of the document (1 based)", ge=1)],
    retrieve_type: Annotated[
        Literal["text", "image", "both"], Field(alias="type", description="Content type to retrieve")
    ] = "both",
) -> list[TextContent | ImageContent]:
    """Returns the full content of specific document's page range (text, image, or both)."""
    chunk_service = await afind_instance(ChunkService)

    if page_start > page_end:
        raise ValueError("'page_start' cannot be greater than 'page_end'")
    requested_pages = page_end - page_start + 1
    if requested_pages > GET_PAGES_LIMIT:
        raise ValueError(
            f"you can request maximum {GET_PAGES_LIMIT} pages in single tool call "
            f"({requested_pages} pages requested)"
        )

    doc_pages = [(document_id, page_idx) for page_idx in range(page_start, page_end + 1)]

    match retrieve_type:
        case "text":
            chunks = await chunk_service.get_chunks_by_pages(*doc_pages, chunk_type=ChunkType.text)
        case "image":
            chunks = await chunk_service.get_chunks_by_pages(*doc_pages, chunk_type=ChunkType.image)
        case "both":
            chunks = await chunk_service.get_chunks_by_pages(*doc_pages)

    text_content: dict[int, TextContent] = {}
    image_content: dict[int, ImageContent] = {}

    for chunk in chunks:
        if isinstance(chunk, TextChunk):
            if content_block := text_content.get(chunk.metadata.page_number):
                content_block.text += "\n" + chunk.text
            else:
                text_content[chunk.metadata.page_number] = TextContent(type="text", text=chunk.text)

        elif isinstance(chunk, ImageChunk) and chunk.image_type == ImageType.page:
            image_content[chunk.metadata.page_number] = ImageContent(
                type="image", data=base64.b64encode(chunk.content).decode(), mimeType=chunk.mime_type
            )

    return list(_get_pages_content(document_id, text_content, image_content))


def _get_pages_content(
    document_id: int, page_text: dict[int, TextContent], page_images: dict[int, ImageContent]
):
    for page_idx in sorted(set(page_text.keys()) | set(page_images.keys())):
        yield TextContent(type="text", text=f"[Document {document_id}, Page {page_idx}]")
        if content_block := page_text.get(page_idx):
            yield content_block
        if content_block := page_images.get(page_idx):
            yield content_block


@provider.tool(annotations=ToolAnnotations(destructiveHint=False))
async def get_citation_url(
    document_ids: Annotated[
        list[Annotated[int, annotated_types.Ge(1)]], Field(description="IDs of required documents")
    ],
) -> dict[int, str]:
    """Share given documents with a user. Returns a mapping of `{id: url}`."""
    document_service = await afind_instance(DocumentService)
    file_storage = await afind_instance(FileStorage)

    documents = await document_service.get_documents_by_id(document_ids)
    async with TaskGroup() as task_group:
        tasks = {
            doc.id: task_group.create_task(
                file_storage.copy_file_to_user(
                    doc.url,
                    doc.display_name,
                )
            )
            for doc in documents
        }
    return {k: v.result() for k, v in tasks.items()}


class RetrievedChunk(BaseModel):
    document_id: Annotated[int, Field(description="`id` of related document")]
    chunk_id: Annotated[int, Field(description="`id` of chunk within the document")]
    text: Annotated[str, Field(description="text of retrieved chunk")]
    page_number: Annotated[int, Field(description="number of page where this chunk was extracted")]
    metadata: Annotated[dict | None, Field(description="metadata of related document", default_factory=dict)]

    @classmethod
    def create(cls, doc: RetrievedDocument, metadata_field_names: set[str] | None = None) -> Iterable[Self]:
        for chunk in doc.chunks:
            if isinstance(chunk, TextChunk):
                yield cls(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    page_number=chunk.metadata.page_number,
                    metadata=(
                        {k: v for k, v in doc.source_metadata.items() if k in metadata_field_names}
                        if metadata_field_names is not None
                        else doc.source_metadata
                    ),
                )
            # NOTE: don't include image chunks -
            # they bloat the context and likely don't bring much value.
            # we expose get_page tool with image mode.


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


@provider.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
async def retrieve_text_chunks(
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
    channel = await afind_instance(Channel)
    metadata_service = await afind_instance(MetadataService)

    request_config_model = await RequestConfig.get_dynamic_model()
    request_config = request_config_model.create(
        defaults=channel.request_config,
        overrides={
            "retriever": _get_retriever_overrides(document_ids, metadata_filter),
        },
    )

    retriever = Retriever.create(request_config.retriever)
    metadata_field_names = metadata_service.get_mcp_retrieve_chunks_field_names()

    return list(
        itertools.chain(*[
            RetrievedChunk.create(doc, metadata_field_names)
            for doc in await retriever.invoke(query, PlainAnswer())
        ])
    )


_SEARCH_QUERY_DESCRIPTION = """\
Natural-language search query — a question phrased the way one person would ask another.

Each query should target a single piece of information.
For compound requests (e.g. asking for a number and a reason), issue multiple calls — one atomic question per call.

Do not include filtering or scoping instructions in the query; use other arguments for that.
"""


@provider.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
async def rag_search(
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
    channel = await afind_instance(Channel)

    request_config_model = await RequestConfig.get_dynamic_model()
    request_config = request_config_model.create(
        defaults=channel.request_config,
        overrides={
            "retriever": _get_retriever_overrides(document_ids, metadata_filter),
            "generation": {
                "type": "default",
            },
        },
    )

    retriever = Retriever.create(request_config.retriever)
    answer_generator = AnswerGenerator.create(request_config.generation)
    answer = PlainAnswer()

    await answer_generator.invoke(query, retriever, answer)

    text = answer.content
    if not answer.has_references:
        text += "\n\nNo references found"
    return TextContent(type="text", text=text)


DOCUMENT_IDS_PATTERN = r"^[1-9][0-9]*(,[1-9][0-9]*)*$"

_DOCUMENT_IDS_DESCRIPTION = """\
IDs of the required documents, joined with commas and nothing else, for example `1,5,9`.
A single id is allowed. A trailing comma, a space after a comma, a non-numeric id or an id of zero
makes the whole read fail, rather than that one id being dropped from the answer silently.
"""


@provider.resource("documents://metadata/{document_ids}", mime_type="application/json")
async def documents_metadata(
    document_ids: Annotated[str, Field(description=_DOCUMENT_IDS_DESCRIPTION, pattern=DOCUMENT_IDS_PATTERN)],
) -> dict[int, dict[str, Any]]:
    """
    Metadata of the requested documents, as a JSON object keyed by document id.

    Each value is that document's metadata exactly as stored: every key the channel holds,
    none of them renamed, and nothing added. The channel's own metadata schema is therefore
    what tells a consumer which key carries the title, the publication date, and so on.

    An id this channel does not know is absent from the answer rather than an error, so a
    caller must not assume every id it asked for comes back.

    This is a resource rather than a tool because application code, not a model, chooses the
    ids and reads it: the answer is the same for every caller within a channel, and the read
    has no side effect.
    """
    document_service = await afind_instance(DocumentService)

    documents = await document_service.get_documents_by_id(
        int(document_id) for document_id in document_ids.split(",")
    )
    return {document.id: document.metadata for document in documents}


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
                list_documents_unordered.__name__,
                retrieve_text_chunks.__name__,
                rag_search.__name__,
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

            if current_tool.name == list_documents_unordered.__name__:
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
