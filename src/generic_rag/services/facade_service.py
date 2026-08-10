from asyncio import TaskGroup

from aidial_sdk.exceptions import ResourceNotFoundError
from fastapi import UploadFile
from injection import scoped

from generic_rag.app.jobs import run_index_document_job
from generic_rag.channel import Channel
from generic_rag.db.session import transaction
from generic_rag.scope import ScopeName
from generic_rag.services.chunk_service import ChunkService
from generic_rag.services.document_service import DocumentService
from generic_rag.services.indexing_service import IndexingService
from generic_rag.types import Document, DocumentStatus


@scoped(ScopeName.channel)
class FacadeService:
    """High-level service for complex scenarios."""

    def __init__(
        self,
        channel: Channel,
        document_service: DocumentService,
        chunk_service: ChunkService,
        indexing_service: IndexingService,
    ):
        self._channel = channel
        self._document_service = document_service
        self._chunk_service = chunk_service
        self._indexing_service = indexing_service

    async def create_document(
        self, attachment: UploadFile, folder: str | None = None, metadata: dict | None = None
    ) -> Document:
        """
        Upload document to a channel.

        :param attachment: the file to upload
        :param folder: path of a target folder within a channel (can have multiple parts)
        :param metadata: metadata to assign with document (should match JSON schema associated with this channel)
        """
        document = await self._document_service.create_document(attachment, folder, metadata)

        if document.status == DocumentStatus.ready:
            return document

        return await self._index_document(document)

    async def update_document(
        self, document_id: int, attachment: UploadFile | None = None, metadata: dict | None = None
    ) -> Document:
        """
        Update document with given ID by replacing its content and/or metadata.

        :param document_id: the id of required document
        :param attachment: the file to replace the document's content with
        :param metadata: metadata to assign with document (should match JSON schema associated with this channel)
        """
        document = await self._document_service.update_document(document_id, attachment, metadata)

        if document.status == DocumentStatus.ready:
            return document

        return await self._index_document(document)

    @transaction
    async def delete_document(self, document_id: int) -> None:
        """
        Delete document with given id (and all related data).

        :param document_id: id of required document
        """
        if not await self._document_service.exists(document_id):
            raise ResourceNotFoundError(f"Document '{document_id}' not found.")

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

    async def _index_document(self, document: Document) -> Document:
        await self._delete_document_data(document.id)

        if await run_index_document_job(document.id):
            return document

        await self._indexing_service.index_document(document)

        return await self._document_service.get_document(document.id)
