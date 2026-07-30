import asyncio
import enum
import hashlib
import logging
from asyncio import CancelledError
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from enum import StrEnum

import asyncpg
from aiohttp import ClientSession
from async_lru import alru_cache
from fastapi import FastAPI, HTTPException
from injection import afind_instance, inject, singleton
from pgqueuer import DatabaseRetryEntrypointExecutor, Job, PgQueuer, Queries
from pgqueuer.adapters.tracing.opentelemetry import OpenTelemetryTracing
from pgqueuer.adapters.web import create_web_router
from pgqueuer.domain.errors import DuplicateJobError
from pgqueuer.ports.tracing import set_tracing_class
from pydantic import BaseModel
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

from generic_rag.app.settings import ApplicationSettings
from generic_rag.scope import DialApplicationId

logger = logging.getLogger(__name__)

set_tracing_class(OpenTelemetryTracing())


@enum.unique
class EntrypointName(StrEnum):
    index_document = "document.index"


@alru_cache(ttl=3600)
async def _check_application_access(application_id: str) -> bool:
    """Check if given application is accessible using global api-key."""
    settings = await afind_instance(ApplicationSettings)

    if not (application_id and settings.dial_api_key):
        return False

    client_session = await afind_instance(ClientSession)
    url = f"{settings.dial_url}/v1/deployments/{application_id}/route/channel/config"

    async with client_session.get(
        url, headers={"api-key": settings.dial_api_key.get_secret_value()}
    ) as response:
        return response.ok


async def run_index_document_job(document_id: int, index_names: set[str] | None = None, force: bool = False):
    """
    Run background job for indexing of given document, if possible.

    :param document_id: the ID of a document to index
    :param index_names: names of indexes to update (if not defined - all indexes will be updated)
    :param force: perform whole processing of a document
    :return: `True` if the job was created, or `False` otherwise
    """
    application_id = await afind_instance(DialApplicationId)

    if not await _check_application_access(application_id):
        return False

    logger.info(f"Document '{document_id}' will be indexed in background")

    queries = await afind_instance(Queries)
    payload = IndexDocumentJobPayload(
        application_id=application_id,
        document_id=document_id,
        index_names=index_names or None,
        force=force,
    )

    try:
        job_ids = await queries.enqueue(
            EntrypointName.index_document,
            payload.model_dump_json().encode(),
            dedupe_key=hashlib.md5(f"{application_id}/{document_id}".encode()).hexdigest(),
        )
    except DuplicateJobError as e:
        logger.warning(f"{e!r}")
        raise HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Job for indexing document {document_id} already created.",
        ) from e

    logger.info(f"Created jobs: {job_ids}")

    return True


class IndexDocumentJobPayload(BaseModel):
    application_id: str
    document_id: int
    index_names: set[str] | None = None
    force: bool = False


async def index_document_entrypoint(job: Job):
    if not job.payload:
        raise RuntimeError("payload cannot be empty!")
    payload = IndexDocumentJobPayload.model_validate_json(job.payload)

    settings = await afind_instance(ApplicationSettings)
    client_session = await afind_instance(ClientSession)

    if not settings.dial_api_key:
        raise RuntimeError("DIAL api-key is not set")

    application_route_url = f"{settings.dial_url}/v1/deployments/{payload.application_id}/route"
    request_url = f"{application_route_url}/channel/documents/{payload.document_id}/reindex"

    params = [("async", "false")]
    if payload.index_names:
        params.extend(("index", name) for name in payload.index_names)
    if payload.force:
        params.append(("force", "true"))

    logger.info(f"indexing document with {payload.document_id=} of '{payload.application_id}'")

    async with client_session.put(
        url=request_url, params=params, headers={"api-key": settings.dial_api_key.get_secret_value()}
    ) as response:
        response.raise_for_status()

    logger.info("done")


@singleton
async def pgqueuer_factory(asyncpg_pool: asyncpg.Pool) -> PgQueuer:
    pgq = PgQueuer.from_asyncpg_pool(asyncpg_pool)
    pgq.entrypoint(
        EntrypointName.index_document,
        executor_factory=lambda params: DatabaseRetryEntrypointExecutor(params, max_attempts=3),
        concurrency_limit=2,
    )(index_document_entrypoint)
    return pgq


@singleton
def queries_factory(pgq: PgQueuer) -> Queries:
    return Queries(pgq.connection)


@asynccontextmanager
@inject
async def _run_worker(pgq: PgQueuer = NotImplemented):
    def _on_task_done(task: asyncio.Task):
        nonlocal worker_task

        if task.cancelled():
            return

        logger.info("Worker finished unexpectedly")
        if exc := task.exception():
            logger.warning(f"{exc}", exc_info=exc)

        pgq.shutdown.clear()

        logger.info("Restarting worker")
        worker_task = asyncio.create_task(pgq.run(shutdown_on_listener_failure=True))
        worker_task.add_done_callback(_on_task_done)

    worker_task = asyncio.create_task(pgq.run(shutdown_on_listener_failure=True))
    worker_task.add_done_callback(_on_task_done)

    logger.info("Worker started")

    try:
        yield
    finally:
        logger.info("Stopping worker")
        worker_task.cancel()
        with suppress(CancelledError):
            await worker_task
        logger.info("Worker finished")


@inject
async def setup_jobs(
    app: FastAPI = NotImplemented,
    queries: Queries = NotImplemented,
    exit_stack: AsyncExitStack = NotImplemented,
):
    await queries.upgrade()
    app.state.pgq_queries = queries
    app.include_router(create_web_router(include_sse=False), prefix="/dashboard")
    await exit_stack.enter_async_context(_run_worker())
