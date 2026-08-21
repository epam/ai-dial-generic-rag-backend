import enum
import hashlib
import logging
from asyncio import TaskGroup
from enum import StrEnum
from typing import Literal

from aidial_sdk.exceptions import InvalidRequestError, ResourceNotFoundError
from fastapi import UploadFile
from injection import afind_instance, scoped
from pgqueuer import Queries

from generic_rag.app.jobs import (
    CreateChannelArchiveJobPayload,
    EntrypointName,
    IndexDocumentJobPayload,
    run_background_job,
)
from generic_rag.channel import Channel
from generic_rag.db.session import transaction
from generic_rag.scope import DialApplicationId, ScopeName
from generic_rag.services.chunk_service import ChunkService
from generic_rag.services.document_service import DocumentService
from generic_rag.services.export_service import ExportService
from generic_rag.services.indexing_service import IndexingService
from generic_rag.types import Document, DocumentStatus, FileMetadata, FileStorage

logger = logging.getLogger(__name__)


@enum.unique
class ChannelArchiveStatus(StrEnum):
    ready = enum.auto()
    pending = enum.auto()
    not_found = enum.auto()
    error = enum.auto()


@scoped(ScopeName.channel)
class FacadeService:
    """High-level service for complex scenarios."""

    def __init__(  # noqa: PLR0913
        self,
        channel: Channel,
        document_service: DocumentService,
        chunk_service: ChunkService,
        indexing_service: IndexingService,
        export_service: ExportService,
        file_storage: FileStorage,
    ):
        self._channel = channel
        self._document_service = document_service
        self._chunk_service = chunk_service
        self._indexing_service = indexing_service
        self._export_service = export_service
        self._file_storage = file_storage

    async def create_document(
        self,
        attachment: UploadFile,
        folder: str | None = None,
        metadata: dict | None = None,
        overwrite: bool = False,
    ) -> Document:
        """
        Upload document to a channel.

        :param attachment: the file to upload
        :param folder: path of a target folder within a channel (can have multiple parts)
        :param overwrite: allow to overwrite the document that already exists (if any)
        :param metadata: metadata to assign with document (should match JSON schema associated with this channel)
        """
        document = await self._document_service.create_document(attachment, folder, metadata, overwrite)
        await self._reset_channel_archive()

        if document.status == DocumentStatus.ready:
            return document

        await self._delete_document_data(document.id)
        document, _ = await self._index_document(document)

        return document

    async def update_document(
        self, document_id: int, attachment: UploadFile | None = None, metadata: dict | None = None
    ) -> Document:
        """
        Update document with given ID by replacing its content and/or metadata.

        :param document_id: the id of target document
        :param attachment: the file to replace the document's content with
        :param metadata: metadata to assign with document (should match JSON schema associated with this channel)
        """
        document = await self._document_service.update_document(document_id, attachment, metadata)
        await self._reset_channel_archive()

        if document.status == DocumentStatus.ready:
            return document

        await self._delete_document_data(document.id)
        document, _ = await self._index_document(document)

        return document

    async def reindex_document(
        self,
        document_id: int,
        index_names: set[str] | None = None,
        force: bool = False,
        background: bool = True,
    ) -> tuple[Document, bool]:
        """
        Index or reindex document with given ID.

        :param document_id: the id of target document
        :param index_names: names of indexes to update (if not defined - all indexes will be updated)
        :param force: perform whole process, including document re-processing and rebuilding of all indexes;
          it not set, document processing will be performed only if the document wasn't processed yet
        :param background: run action via background job (if possible)
        :return: pair of (document,background)
        """
        document = await self._document_service.get_document(document_id)

        return await self._index_document(
            document,
            index_names=index_names,
            force=force,
            mode="auto" if background else "sync",
        )

    @transaction
    async def delete_document(self, document_id: int) -> None:
        """
        Delete document with given id (and all related data).

        :param document_id: id of required document
        """
        if not await self._document_service.exists_by_id(document_id):
            raise ResourceNotFoundError(f"Document '{document_id}' not found.")

        await self._reset_channel_archive()
        await self._delete_document_data(document_id)
        await self._document_service.delete_document(document_id)

    async def _delete_document_data(self, document_id):
        async with TaskGroup() as task_group:
            # cleanup indexes
            for idx in await self._channel.get_indexes():
                task_group.create_task(idx.storage.remove(document_id))

            # remove chunks and their data (we cannot rely only on cascade delete
            # because image chunks has files that wouldn't be deleted in that case)
            task_group.create_task(self._chunk_service.delete_chunks_by_document(document_id))

    async def _index_document(
        self,
        document: Document,
        *,
        index_names: set[str] | None = None,
        force: bool = False,
        mode: Literal["auto", "sync"] = "auto",
    ) -> tuple[Document, bool]:
        """
        Index given document. If the document can be indexed in background - return
        immediately, otherwise will block until the document will be fully indexed.

        :param document: document to index
        :param index_names: names of indexes to update (if not defined - all indexes will be updated)
        :param force: perform whole processing of a document
        """
        if mode != "sync":
            application_id = await afind_instance(DialApplicationId)

            if await run_background_job(
                EntrypointName.index_document,
                payload=IndexDocumentJobPayload(
                    application_id=application_id,
                    document_id=document.id,
                    index_names=index_names or None,
                    force=force,
                ),
                dedupe_key=hashlib.md5(f"{application_id}/{document.id}".encode()).hexdigest(),
            ):
                logger.info(f"Document '{document.id}' will be indexed in background")
                return document, True

        await self._reset_channel_archive()
        await self._indexing_service.index_document(document, index_names=index_names, force=force)

        return await self._document_service.get_document(document.id), False

    async def get_channel_archive(self) -> FileMetadata | None:
        """Get metadata object of channel archive."""
        bucket = await self._file_storage.get_bucket()
        return await self._file_storage.get_file_metadata(
            f"files/{bucket}/export/{self._channel.channel_key}.zip"
        )

    async def get_channel_archive_status(self) -> ChannelArchiveStatus:
        """Return the status the channel archiving process."""
        queries = await afind_instance(Queries)
        application_id = await afind_instance(DialApplicationId)

        for job in await queries.browse_queue(
            entrypoints=[EntrypointName.create_channel_archive], statuses=["queued", "picked"]
        ):
            if not job.payload:
                continue

            payload = CreateChannelArchiveJobPayload.model_validate_json(job.payload)
            if payload.application_id == application_id:
                return ChannelArchiveStatus.pending

        for job in await queries.browse_queue(
            entrypoints=[EntrypointName.create_channel_archive], statuses=["exception"]
        ):
            if not job.payload:
                continue

            payload = CreateChannelArchiveJobPayload.model_validate_json(job.payload)
            if payload.application_id == application_id:
                return ChannelArchiveStatus.error

        if await self.get_channel_archive():
            return ChannelArchiveStatus.ready

        return ChannelArchiveStatus.not_found

    async def create_channel_archive(self, background: bool) -> ChannelArchiveStatus:
        """
        Run the process of channel archive creation.

        :param background: run action via background job (if possible)
        """
        if background:
            if (status := await self.get_channel_archive_status()) in {
                ChannelArchiveStatus.pending,
                ChannelArchiveStatus.ready,
            }:
                return status

            if await run_background_job(
                EntrypointName.create_channel_archive,
                payload=CreateChannelArchiveJobPayload(
                    application_id=await afind_instance(DialApplicationId),
                ),
                dedupe_key=self._channel.channel_key,
            ):
                return ChannelArchiveStatus.pending

            raise InvalidRequestError(message="Could not create channel archive.")

        try:
            await self._export_service.export_channel(f"export/{self._channel.channel_key}.zip")
        except Exception as e:
            logger.warning(str(e))
            return ChannelArchiveStatus.error
        else:
            return ChannelArchiveStatus.ready

    async def _reset_channel_archive(self):
        queries = await afind_instance(Queries)
        application_id = await afind_instance(DialApplicationId)
        jobs_to_cancel = []

        for job in await queries.browse_queue(
            entrypoints=[EntrypointName.create_channel_archive], statuses=["queued", "picked"]
        ):
            if not job.payload:
                continue

            payload = CreateChannelArchiveJobPayload.model_validate_json(job.payload)
            if payload.application_id == application_id:
                jobs_to_cancel.append(job.id)

        if jobs_to_cancel:
            await queries.mark_job_as_cancelled(jobs_to_cancel)

        bucket = await self._file_storage.get_bucket()
        url = f"files/{bucket}/export/{self._channel.channel_key}.zip"

        if await self._file_storage.get_file_metadata(url):
            await self._file_storage.delete_file(url)
