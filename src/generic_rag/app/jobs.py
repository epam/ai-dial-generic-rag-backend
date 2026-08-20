import asyncio
import dataclasses
import enum
import functools
import logging
import signal
from abc import ABC, abstractmethod
from asyncio import CancelledError
from contextlib import AsyncExitStack, suppress
from enum import StrEnum
from types import FrameType
from typing import Literal, overload

import asyncpg
from aiohttp import ClientSession
from async_lru import alru_cache
from fastapi import FastAPI, HTTPException
from injection import afind_instance, inject, singleton
from pgqueuer import DatabaseRetryEntrypointExecutor, Job, PgQueuer, Queries
from pgqueuer.adapters.tracing.opentelemetry import OpenTelemetryTracing
from pgqueuer.adapters.web import create_web_router
from pgqueuer.core.executors import (
    EntrypointExecutor,
    EntrypointExecutorParameters,
)
from pgqueuer.domain.errors import DuplicateJobError
from pgqueuer.domain.models import Context
from pgqueuer.ports.tracing import set_tracing_class
from pydantic import BaseModel
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

from generic_rag.app.settings import ApplicationSettings
from generic_rag.utils.generics import resolve_generic_arg

logger = logging.getLogger(__name__)

set_tracing_class(OpenTelemetryTracing())


class JobPayload(BaseModel):
    application_id: str


class JobRunner[T: JobPayload](ABC):
    """Base class for Job Runner implementations."""

    _payload: T

    @inject
    def __init__(
        self,
        job: Job,
        *,
        settings: ApplicationSettings = NotImplemented,
        client_session: ClientSession = NotImplemented,
    ):
        self._client_session = client_session

        if not settings.dial_api_key:
            raise ValueError("DIAL api-key is not set")

        payload_type = resolve_generic_arg(self, JobRunner, 0)

        if not (payload_type and isinstance(payload_type, type) and issubclass(payload_type, JobPayload)):
            raise ValueError("job payload type is incorrect")
        if not job.payload:
            raise ValueError("payload cannot be empty!")

        self._payload = payload_type.model_validate_json(job.payload)
        self._application_route_url = (
            f"{settings.dial_url}/v1/deployments/{self._payload.application_id}/route"
        )
        self._headers = {"api-key": settings.dial_api_key.get_secret_value()}

    @abstractmethod
    async def run(self, context: Context): ...


class IndexDocumentJobPayload(JobPayload):
    document_id: int
    index_names: set[str] | None = None
    force: bool = False


class IndexDocumentJobRunner(JobRunner[IndexDocumentJobPayload]):
    async def run(self, context: Context):
        request_url = f"{self._application_route_url}/channel/documents/{self._payload.document_id}/reindex"

        params = [("async", "false")]
        if self._payload.index_names:
            params.extend(("index", name) for name in self._payload.index_names)
        if self._payload.force:
            params.append(("force", "true"))

        logger.info(
            f"indexing document with {self._payload.document_id=} of '{self._payload.application_id}'"
        )

        async with self._client_session.put(
            url=request_url, params=params, headers=self._headers
        ) as response:
            response.raise_for_status()

        logger.info("done")


class CreateChannelArchiveJobPayload(JobPayload): ...


class CreateChannelArchiveJobRunner(JobRunner[CreateChannelArchiveJobPayload]):
    async def run(self, context: Context):
        request_url = f"{self._application_route_url}/channel/jobs/archive/create"

        with context.cancellation:
            async with self._client_session.post(
                request_url, headers=self._headers | {"accept": "text/event-stream"}
            ) as response:
                response.raise_for_status()
                assert response.content_type == "text/event-stream"

                async for line in response.content:
                    if message := line.decode().strip():
                        logger.info(message)

            logger.info("done")


class ImportChannelArchiveJobPayload(JobPayload):
    archive_url: str


class ImportChannelArchiveJobRunner(JobRunner[ImportChannelArchiveJobPayload]):
    async def run(self, context: Context):
        request_url = f"{self._application_route_url}/channel/jobs/archive/import"

        with context.cancellation:
            async with self._client_session.post(
                request_url,
                data={"url": self._payload.archive_url},
                headers=self._headers | {"accept": "text/event-stream"},
            ) as response:
                response.raise_for_status()
                assert response.content_type == "text/event-stream"

                async for line in response.content:
                    if message := line.decode().strip():
                        logger.info(message)

            logger.info("done")


@enum.unique
class EntrypointName(StrEnum):
    index_document = "document.index"
    create_channel_archive = "channel.archive.create"
    import_channel_archive = "channel.archive.import"


@overload
async def enqueue_job(
    entrypoint: Literal[EntrypointName.index_document],
    payload: IndexDocumentJobPayload,
    *,
    dedupe_key: str | None = None,
): ...


@overload
async def enqueue_job(
    entrypoint: Literal[EntrypointName.create_channel_archive],
    payload: CreateChannelArchiveJobPayload,
    *,
    dedupe_key: str | None = None,
): ...


@overload
async def enqueue_job(
    entrypoint: Literal[EntrypointName.import_channel_archive],
    payload: ImportChannelArchiveJobPayload,
    *,
    dedupe_key: str | None = None,
): ...


async def enqueue_job[T: JobPayload](
    entrypoint: EntrypointName, payload: T, *, dedupe_key: str | None = None
) -> bool:
    """
    Run given background job, if possible.

    :param entrypoint: name of the job entrypoint
    :param payload: the payload to pass to the job
    :param dedupe_key: optional key used to prevent duplicate jobs from entering the queue
    :return: `True` if the job was created, or `False` otherwise
    """
    if not await _check_application_access(payload.application_id):
        return False

    queries = await afind_instance(Queries)

    try:
        job_ids = await queries.enqueue(
            entrypoint,
            payload.model_dump_json().encode(),
            dedupe_key=dedupe_key,
        )
    except DuplicateJobError as e:
        logger.warning(f"{e!r}")
        raise HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
        ) from e  # todo: return 400 status, because core converts 429 response into 503

    logger.info(f"Created jobs: {job_ids}")

    return True


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


def _executor_factory(params: EntrypointExecutorParameters) -> EntrypointExecutor:
    """
    Create executor for given entrypoint.

    Uses :class:`DatabaseRetryEntrypointExecutor` for actual job execution
    but wraps the entrypoint function with a decorator which converts CancelledError
    into RuntimeError, so that if the job execution was terminated (this might happen
    due timeout on a worker shutdown) it will be rescheduled for another execution
    attempt by :class:`DatabaseRetryEntrypointExecutor`.
    """

    @functools.wraps(params.func)
    async def wrapper(*args, **kwargs):
        try:
            return await params.func(*args, **kwargs)
        except CancelledError as e:
            raise RuntimeError(f"Unexpected cancellation: {str(e)}") from e

    return DatabaseRetryEntrypointExecutor(dataclasses.replace(params, func=wrapper), max_attempts=3)


@singleton
async def pgqueuer_factory(asyncpg_pool: asyncpg.Pool) -> PgQueuer:
    pgq = PgQueuer.from_asyncpg_pool(asyncpg_pool)

    @pgq.entrypoint(EntrypointName.index_document, executor_factory=_executor_factory, concurrency_limit=2)
    async def index_document_entrypoint(job: Job, context: Context):
        await IndexDocumentJobRunner(job).run(context)

    @pgq.entrypoint(
        EntrypointName.create_channel_archive, executor_factory=_executor_factory, concurrency_limit=1
    )
    async def create_channel_archive_entrypoint(job: Job, context: Context):
        await CreateChannelArchiveJobRunner(job).run(context)

    @pgq.entrypoint(
        EntrypointName.import_channel_archive, executor_factory=_executor_factory, concurrency_limit=1
    )
    async def import_channel_archive_entrypoint(job: Job, context: Context):
        await ImportChannelArchiveJobRunner(job).run(context)

    return pgq


@singleton
def queries_factory(pgq: PgQueuer) -> Queries:
    return Queries(pgq.connection)


@inject
async def _worker_main(stop_event: asyncio.Event, pgq: PgQueuer = NotImplemented):
    def _on_task_done(task: asyncio.Task):
        nonlocal worker_task

        if task.cancelled():
            return

        logger.info("Worker task finished unexpectedly")
        if exc := task.exception():
            logger.warning(f"{exc}", exc_info=exc)

        pgq.shutdown.clear()

        logger.info("Restarting worker task")
        worker_task = asyncio.create_task(pgq.run(shutdown_on_listener_failure=True))
        worker_task.add_done_callback(_on_task_done)

    worker_task = asyncio.create_task(pgq.run(shutdown_on_listener_failure=True))
    worker_task.add_done_callback(_on_task_done)

    logger.info("Worker started")

    try:
        while not stop_event.is_set():
            with suppress(TimeoutError):
                await asyncio.wait_for(stop_event.wait(), 30)
    finally:
        logger.info("Stopping worker")
        if worker_task.cancel():
            with suppress(CancelledError, TimeoutError):
                await asyncio.wait_for(worker_task, 15)
        logger.info("Worker finished")


@inject
async def run_worker(exit_stack: AsyncExitStack = NotImplemented):
    stop_event = asyncio.Event()

    def _stop_worker(sig: int, frame: FrameType | None):
        stop_event.set()
        for sig_, handler in original_handlers.items():
            signal.signal(sig_, handler)
        signal.raise_signal(sig)

    original_handlers = {
        sig: signal.signal(sig, _stop_worker)
        for sig in {
            signal.SIGINT,
            signal.SIGTERM,
        }
    }

    exit_stack.push_async_callback(
        functools.partial(
            asyncio.wait_for,
            asyncio.create_task(_worker_main(stop_event)),
            None,
        )
    )


@inject
async def setup_jobs(app: FastAPI, queries: Queries = NotImplemented):
    await queries.upgrade()
    app.state.pgq_queries = queries
    app.include_router(
        create_web_router(include_sse=False),
        prefix="/dashboard",
    )

    await run_worker()
