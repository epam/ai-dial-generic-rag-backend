import enum
import logging
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import NamedTuple

from injection import asfunction, get_instance, singleton
from taskiq import AsyncBroker, InMemoryBroker, TaskiqMessage, TaskiqMiddleware
from taskiq.middlewares.opentelemetry_middleware import OpenTelemetryMiddleware

from generic_rag.scope import DialApplicationId
from generic_rag.services.chunk_service import ChunkService
from generic_rag.services.document_service import DocumentService
from generic_rag.services.indexing_service import IndexingService
from generic_rag.types import DocumentStatus

logger = logging.getLogger(__name__)


@enum.unique
class TaskName(enum.StrEnum):
    index_document = enum.auto()


class DialApplicationMiddleware(TaskiqMiddleware):
    """ Middleware that adds DIAL application id to a message labels. """

    def pre_send(self, message: TaskiqMessage):
        message.labels["dial-application-id"] = get_instance(DialApplicationId, None)

        return super().pre_send(message)


class DocumentStatusUpdateHelper:
    """ Helper class to run document processing with automatic status update upon the processing. """

    def __init__(self, document_id: int, document_service: DocumentService):
        self._document_id = document_id
        self._document_service = document_service

    def begin(self, *, started: DocumentStatus, finished: DocumentStatus) -> AbstractAsyncContextManager:
        """
        Get context manager automatic status update.

        :param started: a status that should be set in the beginning
        :param finished: a status that should be set if the processing has finished without errors
        """

        @asynccontextmanager
        async def _helper():
            await self._document_service.set_document_status(self._document_id, started)
            try:
                yield
            except Exception as e:
                logger.error(str(e))
                await self._document_service.set_document_status(self._document_id, DocumentStatus.error)
                raise e
            else:
                await self._document_service.set_document_status(self._document_id, finished)

        return _helper()


@asfunction
class IndexDocumentTask(NamedTuple):
    """ Indexing of a document. """

    application_id: DialApplicationId
    document_service: DocumentService
    chunk_service: ChunkService
    indexing_service: IndexingService

    async def __call__(self, document_id: int, index_names: set[str] | None = None, force: bool = False):
        """
        :param document_id: the ID of document to index
        :param index_names: names of indexes to update (if not defined - all indexes will be updated)
        :param force: perform whole process, including document re-processing and rebuilding of all indexes;
          it not set, document processing will be performed only if the document wasn't processed yet
        """
        logger.info(f"indexing document with {document_id=} of '{self.application_id}'")

        document = await self.document_service.get_document(
            document_id
        )

        status_helper = DocumentStatusUpdateHelper(document_id, self.document_service)

        if force or document.status not in [DocumentStatus.processed, DocumentStatus.ready]:
            async with status_helper.begin(started=DocumentStatus.processing, finished=DocumentStatus.processed):
                await self.chunk_service.delete_chunks_by_document(document.id)
                await self.chunk_service.add_chunks(
                    self.indexing_service.extract_chunks(document),
                )
                index_names = None  # force rebuild of all indexes

        async with status_helper.begin(started=DocumentStatus.indexing, finished=DocumentStatus.ready):
            await self.indexing_service.index_chunks(
                self.chunk_service.get_chunks_by_document(
                    document.id,
                ),
                index_names=index_names,
            )


@singleton
def broker_factory() -> AsyncBroker:
    broker = InMemoryBroker(
        # DIAL does not provide a mechanism to get access to the application bucket
        # and execute requests to models on behalf of a user who initiated the operation;
        # currently this can be achieved only with per-request api keys but these keys
        # are invalidated once a request processing completes, and we cannot continue using
        # them in background, so all tasks will be executed "in place" until this is not solved
        await_inplace=True,
    ).with_middlewares(
        OpenTelemetryMiddleware(),
        DialApplicationMiddleware(),
    )

    broker.register_task(IndexDocumentTask, task_name=TaskName.index_document)

    return broker
