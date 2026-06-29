import io
import json
import os
from collections.abc import AsyncGenerator, Sequence
from typing import Annotated
from urllib.parse import urljoin

from async_lru import alru_cache
from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, Path, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.security import APIKeyHeader
from injection import inject
from injection.ext.fastapi import Inject
from pydantic import BaseModel, Field, SecretStr, ValidationError, create_model
from starlette.responses import StreamingResponse
from starlette.status import HTTP_201_CREATED, HTTP_204_NO_CONTENT, HTTP_404_NOT_FOUND, HTTP_422_UNPROCESSABLE_CONTENT
from starlette.templating import Jinja2Templates
from taskiq import AsyncBroker, AsyncTaskiqTask

from generic_rag.app import APP_NAME, APP_VERSION
from generic_rag.app.settings import ApplicationSettings
from generic_rag.app.tasks import TaskName
from generic_rag.channel import METADATA_SCHEMA_EXAMPLE, Channel
from generic_rag.scope import ChannelBindings
from generic_rag.services.channel_service import ChannelService
from generic_rag.services.document_service import DocumentService
from generic_rag.services.export_service import ExportService
from generic_rag.services.metadata_service import MetadataService
from generic_rag.services.retrieval_service import RetrievalRequest, RetrievalResult, RetrievalService
from generic_rag.types import Document
from generic_rag.utils.pagination import PaginatedResults, Pagination

_channel = APIRouter()


async def _setup_channel_scope(
    api_key: Annotated[str, Depends(
        APIKeyHeader(
            name="Api-Key",
            scheme_name="Api-Key",
            description="Authorization with DIAL api key"
        ))
    ],
    dial_application_id: Annotated[str | None, Header(
        alias="x-dial-application-id",
        include_in_schema=False
    )] = None,
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
    """ Run retriever and return chunks which are relevant to a given query. """
    try:
        return await retrieval_service.data_retrieval(request)
    except ValidationError as e:
        raise RequestValidationError(errors=e.errors()) from e


@_channel.get("/retrieval/schema", tags=["retrieval"])
async def retrieval_request_schema(retrieval_service: Inject[RetrievalService]):
    """ Get JSON-schema of retrieval request body. """
    return await retrieval_service.get_request_schema()


@_channel.get("/documents", tags=["documents"])
async def list_documents(
    pagination: Annotated[Pagination, Depends(get_pagination)],
    document_service: Inject[DocumentService],
) -> PaginatedResults[
    Document
]:
    """ List all documents uploaded to the channel. """
    return await document_service.list_documents(pagination)


@_channel.post("/documents", tags=["documents"], status_code=HTTP_201_CREATED)
async def upload_document(
    attachment: Annotated[UploadFile, File(description="the document to upload")],
    folder: Annotated[str, Query(description="optional folder where to store the file")] = "",
    metadata: Annotated[
        str | None,
        Form(
            description="metadata to assign with document (should match json schema associated with this channel)",
            examples=["{}"],
        )
    ] = None,
    document_service: Inject[DocumentService] = NotImplemented,
    broker: Inject[AsyncBroker] = NotImplemented,
) -> Document:
    """
    Upload document into the channel.

    The value of `metadata` should be a valid json and should match the json-schema associated with this channel.
    """
    if metadata:
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"{metadata}: invalid json: {str(e)}",
            ) from e
    else:
        metadata = None

    document = await document_service.upload_document(folder, attachment, metadata)

    task: AsyncTaskiqTask = await broker.find_task(
        TaskName.index_document
    ).kiq(
        document_id=document.id,
    )

    if await task.is_ready():
        return await document_service.get_document(document.id)

    return document


@_channel.post("/documents/import", tags=["documents"], status_code=HTTP_201_CREATED)
async def import_document(
    attachment: Annotated[UploadFile, File(..., description="the exported document to import")],
    export_service: Inject[ExportService],
) -> Document:
    """ Import document into the channel. """
    return await export_service.import_document(
        await attachment.read()
    )


@_channel.get("/documents/{id}", tags=["documents"])
async def get_document(
    document_id: Annotated[int, Path(alias="id", description="id of the document")],
    document_service: Inject[DocumentService],
) -> Document:
    """ Get document with given id """
    return await document_service.get_document(document_id)


@_channel.delete("/documents/{id}", tags=["documents"], status_code=HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: Annotated[int, Path(alias="id", description="id of the document")],
    document_service: Inject[DocumentService],
):
    """ Delete document from the channel """
    return await document_service.delete_document(
        document_id=document_id
    )


@_channel.get("/documents/{id}/download", tags=["documents"], response_class=StreamingResponse)
async def download_document_content(
    document_id: Annotated[int, Path(alias="id", description="id of the document")],
    document_service: Inject[DocumentService],
):
    """ Download content of the document with given id """
    document = await document_service.get_document(document_id)
    content_stream = await document.get_content_stream()

    if content_stream is None:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"Unable to download document '{document_id}'.",
        )

    return StreamingResponse(
        content=content_stream,
        media_type=document.mime_type,
        headers={
            "Content-Length": str(document.size),
            "Content-Disposition": f'attachment; filename="{os.path.basename(document.display_name)}"',
            "Access-Control-Expose-Headers": "content-disposition",
        }
    )


@_channel.get("/documents/{id}/export", tags=["documents"], response_class=StreamingResponse)
async def export_document_data(
    document_id: Annotated[int, Path(alias="id", description="id of the document")],
    document_service: Inject[DocumentService],
    export_service: Inject[ExportService],
):
    """ Export document and all its indexes. """
    document = await document_service.get_document(document_id)
    document_data = await export_service.export_document(document)

    async def _content_stream() -> AsyncGenerator[bytes]:
        stream = io.BytesIO(document_data)
        while chunk := stream.read(512 * 1024):
            yield chunk

    name, _ = os.path.splitext(os.path.basename(document.display_name))
    return StreamingResponse(
        content=_content_stream(),
        media_type="application/vnd.msgpack",
        headers={
            "Content-Length": str(len(document_data)),
            "Content-Disposition": f'attachment; filename="{name}.msgpack"',
            "Access-Control-Expose-Headers": "content-disposition",
        }
    )


@_channel.put("/documents/{id}/reindex", tags=["documents"])
async def reindex_document(
    document_id: Annotated[int, Path(alias="id", description="id of the document")],
    index_names: Annotated[set[str], Query(
        alias="index",
        default_factory=set,
        description="names of indexes to update (if not defined - all indexes will be updated)"
    )],
    force: Annotated[bool, Query(
        description=(
            "perform whole process, including document re-processing and rebuilding of all indexes "
            "(in this case `index_names` parameter will be ignored); it not set, document processing "
            "will be performed only if the document wasn't processed yet"
        )
    )] = False,
    document_service: Inject[DocumentService] = NotImplemented,
    broker: Inject[AsyncBroker] = NotImplemented,
) -> Document:
    """  Reindex the document with given id. """
    document = await document_service.get_document(
        document_id
    )

    task: AsyncTaskiqTask = await broker.find_task(
        TaskName.index_document
    ).kiq(
        document_id=document.id,
        index_names=index_names or None,
        force=force,
    )

    if await task.is_ready():
        return await document_service.get_document(document_id)

    return document


class MetadataSchemaResponse(BaseModel):
    schema_: Annotated[
        dict,
        Field(
            serialization_alias="schema",
            description="json schema of metadata that can assigned with documents",
            examples=[METADATA_SCHEMA_EXAMPLE],
        )
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
    """ Get the schema of metadata and available filtering dimensions. """
    return MetadataSchemaResponse(
        schema_=channel.metadata_schema,
        dimensions=await metadata_service.get_filtering_dimensions(),
    )


class ChannelConfigMixin(BaseModel):
    """ Additional properties for a channel. """
    channel_key: str = Field(..., description="Unique key associated with this channel.")


@inject
async def _get_channel_router(channel_service: ChannelService) -> APIRouter:
    router = APIRouter(prefix="/channel", dependencies=[Depends(_setup_channel_scope)])
    channel_config_model = await channel_service.get_channel_config_model()

    # noinspection PyTypeChecker
    channel_config_response_model = create_model(
        channel_config_model.__name__,
        __doc__=channel_config_model.__doc__,
        __base__=(channel_config_model, ChannelConfigMixin),
    )

    @router.get("/config", tags=["config"], response_model_exclude_unset=True)
    async def channel_configuration(channel: Inject[Channel]) -> channel_config_response_model:
        """ Configuration of this channel. """
        return channel_config_response_model.model_validate(
            channel.dump_config()
        )

    router.include_router(_channel)

    return router


@inject
def _get_channel_openapi(router: APIRouter, settings: ApplicationSettings):
    with open(os.path.join(os.path.dirname(__file__), "channel.md")) as fp:
        description = fp.read()

    server_url = settings.dial_public_url or settings.dial_url

    return get_openapi(
        title=APP_NAME,
        version=APP_VERSION,
        summary="A channel-specific API of generic-rag application.",
        description=description,
        servers=[{
            "url": urljoin(
                server_url.encoded_string(),
                "/v1/deployments/{application_id}/route"
            ),
            "description": "DIAL application route",
            "variables": {
                "application_id": {
                    "default": "generic-rag-example",
                    "description": "id of the application in DIAL",
                }
            }
        }],
        routes=router.routes,
    )


async def setup_routes(app: FastAPI):
    channel_router = await _get_channel_router()
    templates = Jinja2Templates(
        os.path.join(os.path.dirname(__file__), "templates")
    )

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
                "urls": [
                    {"name": "channel", "url": "/openapi/channel"}
                ],
            }
        )

    app.include_router(channel_router)
