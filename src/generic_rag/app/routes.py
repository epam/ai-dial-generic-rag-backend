import asyncio
import json
import os
from asyncio import CancelledError
from collections.abc import AsyncGenerator, Sequence
from contextlib import suppress
from io import BytesIO
from pathlib import PosixPath
from typing import Annotated, Any, Literal
from urllib.parse import quote, urljoin

from async_lru import alru_cache
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.security import APIKeyHeader
from fastapi.sse import EventSourceResponse, format_sse_event
from injection import inject
from injection.ext.fastapi import Inject
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    SecretStr,
    ValidationError,
    create_model,
    field_validator,
)
from starlette.responses import StreamingResponse
from starlette.status import (
    HTTP_200_OK,
    HTTP_201_CREATED,
    HTTP_202_ACCEPTED,
    HTTP_204_NO_CONTENT,
    HTTP_404_NOT_FOUND,
    HTTP_422_UNPROCESSABLE_CONTENT,
)
from starlette.templating import Jinja2Templates

from generic_rag.app import APP_NAME, APP_VERSION
from generic_rag.app.settings import ApplicationSettings
from generic_rag.channel import METADATA_SCHEMA_EXAMPLE, Channel
from generic_rag.scope import ChannelBindings, DialApplicationId
from generic_rag.services.channel_service import ChannelService
from generic_rag.services.document_matcher import DocumentMatcherConfig
from generic_rag.services.document_service import DocumentService
from generic_rag.services.export_service import ExportService
from generic_rag.services.facade_service import ChannelArchiveStatus, FacadeService
from generic_rag.services.metadata_service import MetadataService
from generic_rag.services.retrieval_service import RetrievalRequest, RetrievalResult, RetrievalService
from generic_rag.types import Document, FileStorage
from generic_rag.utils.pagination import PaginatedResults, Pagination

_channel = APIRouter()


async def _setup_channel_scope(
    api_key: Annotated[
        str,
        Depends(
            APIKeyHeader(name="Api-Key", scheme_name="Api-Key", description="Authorization with DIAL api key")
        ),
    ],
    dial_application_id: Annotated[
        str | None, Header(alias="x-dial-application-id", include_in_schema=False)
    ] = None,
):
    async with ChannelBindings(SecretStr(api_key), dial_application_id).scope.adefine():
        yield


def get_pagination(
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=0)] = 25,
) -> Pagination:
    return Pagination(offset=offset, limit=limit)


@_channel.post("/retrieval", tags=["retrieval"])
async def data_retrieval(
    request: RetrievalRequest,
    retrieval_service: Inject[RetrievalService],
) -> Sequence[RetrievalResult]:
    """Run retriever and return chunks which are relevant to a given query."""
    try:
        return await retrieval_service.data_retrieval(request)
    except ValidationError as e:
        raise RequestValidationError(errors=e.errors()) from e


@_channel.get("/retrieval/schema", tags=["retrieval"])
async def retrieval_request_schema(retrieval_service: Inject[RetrievalService]):
    """Get JSON-schema of retrieval request body."""
    return await retrieval_service.get_request_schema()


@_channel.get("/documents", tags=["documents"])
async def list_documents(
    pagination: Annotated[Pagination, Depends(get_pagination)],
    document_service: Inject[DocumentService],
) -> PaginatedResults[Document]:
    """List all documents uploaded to the channel."""
    return await document_service.list_documents(pagination)


@_channel.post("/documents", tags=["documents"], status_code=HTTP_201_CREATED)
async def create_document(
    attachment: Annotated[UploadFile, File(description="the file to upload")],
    folder: Annotated[str, Query(description="optional folder where to store the file")] = "",
    metadata: Annotated[
        dict[str, Any] | None,
        BeforeValidator(json.loads),
        Form(
            media_type="application/json",
            description="metadata to assign with document (should match json schema associated with this channel)",
            examples=["{}"],
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        Query(description="allow to overwrite the document that already exists (if any)"),
    ] = False,
    facade_service: Inject[FacadeService] = NotImplemented,
) -> Document:
    """
    Upload document into the channel.

    The value of `metadata` should be a valid JSON and should match the json-schema associated with this channel.
    """
    return await facade_service.create_document(attachment, folder, metadata, overwrite)


@_channel.post("/documents/import", tags=["documents"], status_code=HTTP_201_CREATED)
async def import_document(
    attachment: Annotated[UploadFile, File(..., description="the exported document to import")],
    overwrite: Annotated[
        bool,
        Query(description="allow to overwrite the document that already exists (if any)"),
    ] = False,
    export_service: Inject[ExportService] = NotImplemented,
) -> Document:
    """Import document into the channel."""
    return await export_service.import_document(attachment.file, overwrite)


class ExistsResponse(BaseModel):
    exists: bool


@_channel.get("/documents/exists", tags=["documents"])
async def check_document_existence(
    filename: Annotated[str, Query(description="the filename")],
    folder: Annotated[str, Query(description="folder where to store the file")] = "",
    document_service: Inject[DocumentService] = NotImplemented,
) -> ExistsResponse:
    """Check existence of a document by its filename and folder."""
    return ExistsResponse(
        exists=await document_service.exists_by_name(filename, folder),
    )


class DocumentSearchRequest[IndexNameT: str = str, MatcherT: DocumentMatcherConfig = DocumentMatcherConfig](
    BaseModel
):
    query: str = Field(description="The search query.")
    limit: int = Field(5, description="Maximum number of results to return.")
    indexes: list[IndexNameT] | None = Field(
        default=None,
        min_length=1,
        description=(
            "List of document indexes to use, also defines the order of intermediate search stages. "
            "If omitted, indexes with `include_in_hybrid` set to `true` will be used."
        ),
    )
    matcher: MatcherT | None = Field(
        default=None,
        description=(
            "Configuration of document matcher that restricts "
            "search scope to subset of documents that match given criteria."
        ),
    )

    @classmethod
    @inject
    async def get_dynamic_model(cls, channel: Channel = NotImplemented) -> type["DocumentSearchRequest"]:
        index_names = [idx.index_name for idx in await channel.get_indexes("document")]

        # noinspection type-hints
        index_name_model = Literal[tuple(index_names)] if index_names else str
        matcher_model = await DocumentMatcherConfig.get_dynamic_model()

        # noinspection bad-return
        return create_model(
            cls.__name__,
            __base__=cls[index_name_model, matcher_model],
            __doc__=cls.__doc__,
        )

    @field_validator("indexes", mode="after")
    @classmethod
    def ensure_indexes_unique(cls, value: list[IndexNameT] | None):
        if value and len(value) != len(set(value)):
            raise ValueError("Values must be unique.")
        return value


@_channel.post(
    "/documents/search",
    tags=["documents"],
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": DocumentSearchRequest.model_json_schema(
                        ref_template="#/components/schemas/{model}"
                    ),
                }
            }
        }
    },
)
async def search_for_relevant_documents(
    raw_body: dict[str, Any],
    document_service: Inject[DocumentService] = NotImplemented,
) -> Sequence[Document]:
    """
    Search for documents relevant to a given query using indexes.

    The schema of the request body is dynamic and depends on configuration of the channel,
    and here you can see only overall structure of the schema.

    To get the full schema call `GET /channel/documents/search/schema`.
    """
    try:
        model = await DocumentSearchRequest.get_dynamic_model()
        body = model.model_validate(raw_body)
    except ValidationError as e:
        raise RequestValidationError(errors=e.errors()) from e

    return await document_service.search(
        body.query, body.limit, index_names=body.indexes, matcher_config=body.matcher
    )


@_channel.get("/documents/search/schema", tags=["documents"])
async def document_search_request_schema() -> dict[str, Any]:
    """Return dynamic JSON schema of search request body."""
    model = await DocumentSearchRequest.get_dynamic_model()
    return model.model_json_schema()


@_channel.get("/documents/{id}", tags=["documents"])
async def get_document(
    document_id: Annotated[int, Path(alias="id", description="id of the document")],
    document_service: Inject[DocumentService],
) -> Document:
    """Get document with given id"""
    return await document_service.get_document(document_id)


@_channel.put("/documents/{id}", tags=["documents"])
async def update_document(
    document_id: Annotated[int, Path(alias="id", description="id of the document")],
    attachment: Annotated[
        UploadFile | None, File(description="the file to replace the content of the document with")
    ] = None,
    metadata: Annotated[
        dict[str, Any] | None,
        BeforeValidator(json.loads),
        Form(
            media_type="application/json",
            description="metadata to set (should match json schema associated with this channel)",
            examples=["{}"],
        ),
    ] = None,
    facade_service: Inject[FacadeService] = NotImplemented,
) -> Document:
    """Update document with given ID by replacing its content and/or metadata."""
    return await facade_service.update_document(document_id, attachment, metadata)


@_channel.delete("/documents/{id}", tags=["documents"], status_code=HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: Annotated[int, Path(alias="id", description="id of the document")],
    facade_service: Inject[FacadeService],
):
    """Delete document from the channel"""
    return await facade_service.delete_document(document_id)


@_channel.get("/documents/{id}/download", tags=["documents"], response_class=StreamingResponse)
async def download_document_content(
    document_id: Annotated[int, Path(alias="id", description="id of the document")],
    document_service: Inject[DocumentService],
):
    """Download content of the document with given id"""
    document = await document_service.get_document(document_id)
    content_stream = await document.get_content_stream()

    filename = quote(os.path.basename(document.display_name))

    return StreamingResponse(
        content=content_stream,
        media_type=document.mime_type,
        headers={
            "Content-Length": str(document.size),
            "Content-Disposition": f"attachment; filename*=utf-8''{filename}",
            "Access-Control-Expose-Headers": "content-disposition",
        },
    )


@_channel.get("/documents/{id}/export", tags=["documents"], response_class=StreamingResponse)
async def export_document_data(
    document_id: Annotated[int, Path(alias="id", description="id of the document")],
    document_service: Inject[DocumentService],
    export_service: Inject[ExportService],
):
    """Export document and all its indexes."""
    document = await document_service.get_document(document_id)
    stream = BytesIO()

    await export_service.export_document(document, stream)

    async def _content_stream() -> AsyncGenerator[bytes]:
        stream.seek(0)
        while chunk := stream.read(512 * 1024):
            yield chunk

    name, _ = os.path.splitext(os.path.basename(document.display_name))
    filename = quote(f"{document_id}_{name}.msgpack")

    return StreamingResponse(
        content=_content_stream(),
        media_type="application/vnd.msgpack",
        headers={
            "Content-Length": str(stream.tell()),
            "Content-Disposition": f"attachment; filename*=utf-8''{filename}",
            "Access-Control-Expose-Headers": "content-disposition",
        },
    )


@_channel.put("/documents/{id}/metadata", tags=["documents"], deprecated=True)
async def set_document_metadata(
    document_id: Annotated[int, Path(alias="id", description="id of the document")],
    body: Annotated[dict[str, Any], Field(description="the metadata to set")],
    document_service: Inject[DocumentService],
) -> Document:
    """
    Set the metadata for the document.
    Metadata object should match the JSON schema configured for the channel.

    **DEPRECATED**: use `PUT /channel/documents/{id}` instead.
    """
    return await document_service.update_document(document_id, metadata=body)


@_channel.put(
    "/documents/{id}/reindex",
    tags=["documents"],
    responses={
        HTTP_200_OK: {"model": Document, "description": "Document successfully indexed."},
        HTTP_202_ACCEPTED: {"model": Document, "description": "Document will be indexed in background."},
    },
)
async def reindex_document(  # noqa: PLR0913
    response: Response,
    document_id: Annotated[int, Path(alias="id", description="id of the document")],
    index_names: Annotated[
        set[str],
        Query(
            alias="index",
            default_factory=set,
            description="names of indexes to update (if not defined - all indexes will be updated)",
        ),
    ],
    force: Annotated[
        bool,
        Query(
            description=(
                "perform whole process, including document re-processing and rebuilding of all indexes "
                "(in this case `index_names` parameter will be ignored); it not set, document processing "
                "will be performed only if the document wasn't processed yet"
            )
        ),
    ] = False,
    async_: Annotated[bool, Query(alias="async", include_in_schema=False)] = True,
    facade_service: Inject[FacadeService] = NotImplemented,
):
    """Reindex the document with given id."""
    document, background = await facade_service.reindex_document(
        document_id, index_names or None, force, async_
    )
    if background:
        response.status_code = HTTP_202_ACCEPTED
    return document


class MetadataSchemaResponse(BaseModel):
    schema_: Annotated[
        dict,
        Field(
            serialization_alias="schema",
            description="json schema of metadata that can assigned with documents",
            examples=[METADATA_SCHEMA_EXAMPLE],
        ),
    ]
    dimensions: Annotated[
        dict[str, list[str]],
        Field(..., description="list of dimensions and their values that can be used to filter documents"),
    ]


@_channel.get("/metadata", tags=["metadata"])
async def get_metadata_schema(
    channel: Inject[Channel],
    metadata_service: Inject[MetadataService],
) -> MetadataSchemaResponse:
    """Get the schema of metadata and available filtering dimensions."""
    return MetadataSchemaResponse(
        schema_=channel.metadata_schema,
        dimensions=await metadata_service.get_filtering_dimensions(),
    )


class ChannelArchiveResponse(BaseModel):
    status: ChannelArchiveStatus


@_channel.get("/export", tags=["export"], response_class=StreamingResponse)
async def download_channel_content(
    facade_service: Inject[FacadeService],
    application_id: Inject[DialApplicationId],
    file_storage: Inject[FileStorage],
):
    """Download exported data of the channel as single archive."""
    if (archive := await facade_service.get_channel_archive()) is None:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail="Channel archive is not ready.",
        )

    archive_date = archive.updated_at.date().isoformat()
    _, ext = os.path.splitext(archive.name)

    filename = quote(f"{PosixPath(application_id).name}_{archive_date}{ext}")
    content_stream = await file_storage.download_file(archive.url)

    assert content_stream is not None

    return StreamingResponse(
        content=content_stream,
        media_type=archive.content_type,
        headers={
            "Content-Length": str(archive.content_length),
            "Content-Disposition": f"attachment; filename*=utf-8''{filename}",
            "Access-Control-Expose-Headers": "content-disposition",
        },
    )


@_channel.put(
    "/export", tags=["export"], status_code=HTTP_202_ACCEPTED, response_model=ChannelArchiveResponse
)
async def trigger_channel_export(
    request: Request,
    async_: Annotated[bool, Query(alias="async", include_in_schema=False)] = True,
    facade_service: Inject[FacadeService] = NotImplemented,
):
    """Trigger exporting the data of the channel."""
    if async_:
        return ChannelArchiveResponse(
            status=await facade_service.create_channel_archive(True),
        )

    async def _content():
        task = asyncio.create_task(facade_service.create_channel_archive(False))
        try:
            while not task.done():
                yield format_sse_event(comment="ping")
                with suppress(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(task), timeout=10)
                if await request.is_disconnected():
                    break
            yield format_sse_event(
                data_str=ChannelArchiveResponse(status=task.result()).model_dump_json(),
            )
        except CancelledError:
            if task.cancel():
                with suppress(CancelledError):
                    await task
            raise

    return EventSourceResponse(status_code=HTTP_202_ACCEPTED, content=_content())


@_channel.get("/export/status", tags=["export"])
async def get_channel_export_status(facade_service: Inject[FacadeService]) -> ChannelArchiveResponse:
    """Get the status of the archive with exported channel data."""
    return ChannelArchiveResponse(
        status=await facade_service.get_channel_archive_status(),
    )


class ChannelConfigMixin(BaseModel):
    """Additional properties for a channel."""

    channel_key: str = Field(..., description="Unique key associated with this channel.")


@inject
async def _get_channel_router(channel_service: ChannelService = NotImplemented) -> APIRouter:
    router = APIRouter(
        prefix="/channel",
        dependencies=[Depends(_setup_channel_scope)],
        responses={
            HTTP_422_UNPROCESSABLE_CONTENT: {},
        },
    )
    channel_config_model = await channel_service.get_channel_config_model()

    # noinspection PyTypeChecker
    channel_config_response_model = create_model(
        channel_config_model.__name__,
        __doc__=channel_config_model.__doc__,
        __base__=(channel_config_model, ChannelConfigMixin),
    )

    @router.get("/config", tags=["config"], response_model_exclude_unset=True)
    async def channel_configuration(channel: Inject[Channel]) -> channel_config_response_model:
        """Configuration of this channel."""
        return channel_config_response_model.model_validate(channel.dump_config())

    router.include_router(_channel)

    return router


def _process_nested_defs(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    global_schemas = openapi_schema.setdefault("components", {}).setdefault("schemas", "")

    # if we define complex schema in openapi_extra, this will have $defs;
    # here we move these $defs into global dictionary
    for methods in openapi_schema.get("paths", {}).values():
        for content in methods.values():
            request_body = content.get("requestBody", {})
            json_content = request_body.get("content", {}).get("application/json", {})
            schema = json_content.get("schema", {})

            if "$defs" in schema:
                defs = schema.pop("$defs")
                for model_name, model_schema in defs.items():
                    if model_name not in global_schemas:
                        # todo: make sure there is no model_name in global_schemas already,
                        #  otherwise rename the model (and its references) before adding
                        global_schemas[model_name] = model_schema

    return openapi_schema


@inject
def _get_channel_openapi(router: APIRouter, settings: ApplicationSettings):
    with open(os.path.join(str(os.path.dirname(__file__)), "channel.md")) as fp:
        description = fp.read()

    server_url = settings.dial_public_url or settings.dial_url

    openapi_schema = get_openapi(
        title=APP_NAME,
        version=APP_VERSION,
        summary="A channel-specific API of generic-rag application.",
        description=description,
        servers=[
            {
                "url": urljoin(server_url.encoded_string(), "/v1/deployments/{application_id}/route"),
                "description": "DIAL application route",
                "variables": {
                    "application_id": {
                        "default": "generic-rag-example",
                        "description": "id of the application in DIAL",
                    }
                },
            }
        ],
        routes=router.routes,
    )

    return _process_nested_defs(openapi_schema)


async def setup_routes(app: FastAPI):
    channel_router = await _get_channel_router()
    templates = Jinja2Templates(os.path.join(os.path.dirname(__file__), "templates"))

    @app.get("/application-type-schema")
    async def application_schema(channel_service: Inject[ChannelService]):
        model = await channel_service.get_channel_config_model()
        return model.model_json_schema()

    @app.get("/openapi/channel", include_in_schema=False)
    @alru_cache
    async def channel_openapi():
        return _get_channel_openapi(channel_router)

    @app.get("/docs", include_in_schema=False)
    async def docs(request: Request):
        return templates.TemplateResponse(
            name="swagger.j2",
            request=request,
            context={
                "title": f"{APP_NAME} - swagger",
                "urls": [{"name": "channel", "url": "/openapi/channel"}],
            },
        )

    app.include_router(channel_router)
