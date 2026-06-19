import http
import logging
from contextlib import AsyncExitStack, asynccontextmanager

import fastapi
from aidial_sdk import DIALApp
from aidial_sdk.telemetry.types import MetricsConfig, TelemetryConfig, TracingConfig
from aidial_sdk.utils.json import remove_nones
from fastapi import Request
from injection import find_instance, set_constant
from injection.loaders import load_packages
from starlette.responses import JSONResponse

import generic_rag.app.module
import generic_rag.components
import generic_rag.services
from generic_rag.app import APP_NAME
from generic_rag.app.chat_completion import ChannelCompletion
from generic_rag.app.embeddings import EmbeddingsEndpoint
from generic_rag.app.mcp import setup_mcp
from generic_rag.app.routes import setup_routes
from generic_rag.app.settings import ApplicationSettings, get_app_settings
from generic_rag.db.connection import get_engine
from generic_rag.db.session import DbSessionMaker

logger = logging.getLogger(__name__)


# noinspection PyUnusedLocal
def _fastapi_http_exception_handler(request: Request, exc: Exception):
    assert isinstance(exc, fastapi.HTTPException)

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": remove_nones(
                {
                    "message": http.HTTPStatus(exc.status_code).phrase,
                    "type": "runtime_error",
                    "code": exc.status_code,
                    "display_message": exc.detail,
                }
            )
        },
        headers=exc.headers,
    )


@asynccontextmanager
async def lifespan(app: DIALApp):
    settings = find_instance(ApplicationSettings)

    load_packages(
        generic_rag.components,
        generic_rag.services,
    )

    # noinspection PyAbstractClass
    async with AsyncExitStack() as exit_stack:
        set_constant(exit_stack)

        engine = await get_engine(settings.database, exit_stack)
        DbSessionMaker.configure(bind=engine)

        await setup_routes(app)

        app.add_chat_completion(APP_NAME, ChannelCompletion())
        app.add_embeddings(f"{APP_NAME}-embeddings", EmbeddingsEndpoint())

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

    app.add_exception_handler(fastapi.HTTPException, _fastapi_http_exception_handler)

    return app
