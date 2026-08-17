import asyncio
import http
import logging
from contextlib import AsyncExitStack, asynccontextmanager

import aidial_sdk
import fastapi
from aidial_sdk import DIALApp
from aidial_sdk.telemetry.types import MetricsConfig, TelemetryConfig, TracingConfig
from fastapi.encoders import jsonable_encoder
from injection import afind_instance, find_instance, set_constant
from injection.loaders import load_packages
from sqlalchemy.ext.asyncio import AsyncEngine

import generic_rag.app.dependencies
import generic_rag.components
import generic_rag.services
from generic_rag.app import APP_NAME
from generic_rag.app.chat_completion import ChannelCompletion
from generic_rag.app.embeddings import EmbeddingsEndpoint
from generic_rag.app.jobs import setup_jobs
from generic_rag.app.mcp import setup_mcp
from generic_rag.app.routes import setup_routes
from generic_rag.app.settings import ApplicationSettings, get_app_settings
from generic_rag.db import apply_migrations
from generic_rag.db.session import DbSessionMaker

logger = logging.getLogger(__name__)


def _convert_exception(exc: Exception) -> aidial_sdk.HTTPException:
    match exc:
        case aidial_sdk.HTTPException():
            return exc
        case fastapi.HTTPException():
            return aidial_sdk.HTTPException(
                message=http.HTTPStatus(exc.status_code).phrase,
                status_code=exc.status_code,
                display_message=exc.detail,
            )
        case fastapi.exceptions.RequestValidationError():
            return aidial_sdk.exceptions.RequestValidationError(
                "Request validation error", detail=jsonable_encoder(exc.errors())
            )
    return aidial_sdk.HTTPException(
        message=str(exc),
        display_message="Internal error",
    )


@asynccontextmanager
async def lifespan(app: DIALApp):
    settings = find_instance(ApplicationSettings)

    load_packages(
        generic_rag.components,
        generic_rag.services,
    )

    async with AsyncExitStack() as exit_stack:
        set_constant(exit_stack)

        DbSessionMaker.configure(bind=await afind_instance(AsyncEngine))

        await asyncio.to_thread(apply_migrations, settings.database)
        await setup_routes(app)

        app.add_chat_completion(
            APP_NAME,
            ChannelCompletion(
                enable_debug_stages=settings.enable_debug_stages,
            ),
        )
        app.add_embeddings(f"{APP_NAME}-embeddings", EmbeddingsEndpoint())

        await setup_jobs(app)
        await setup_mcp(app, exit_stack)

        yield


def create_app() -> DIALApp:
    settings = set_constant(get_app_settings())

    logger.info(f"Application settings: {settings.model_dump_json(indent=2, exclude_none=True)}")

    app = DIALApp(
        dial_url=settings.dial_url.encoded_string(),
        add_healthcheck=True,
        propagate_auth_headers=False,
        telemetry_config=TelemetryConfig(
            service_name=APP_NAME,
            tracing=TracingConfig(
                logging=True,
            ),
            metrics=MetricsConfig(),
        ),
        lifespan=lifespan,
        openapi_url=None,  # disable automatic documentation
    )

    for cls in [
        Exception,
        fastapi.HTTPException,
        fastapi.exceptions.RequestValidationError,
    ]:
        app.add_exception_handler(cls, lambda request, exc: _convert_exception(exc).to_fastapi_response())

    return app
