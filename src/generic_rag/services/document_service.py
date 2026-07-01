import asyncio
import hashlib
import logging
import os
from collections.abc import AsyncIterable, Awaitable, Callable, Iterable, Sequence
from typing import Self
from urllib.parse import unquote

import jsonschema
from fastapi import HTTPException, UploadFile
from injection import scoped
from pydantic import Field
from sqlalchemy import and_, func, select, update
from sqlalchemy.exc import IntegrityError
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND, HTTP_422_UNPROCESSABLE_CONTENT
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from generic_rag.channel import Channel
from generic_rag.db.entities import DocumentEntity
from generic_rag.db.session import get_current_session, transaction
from generic_rag.scope import ScopeName
from generic_rag.services.chunk_service import ChunkService
from generic_rag.services.document_matcher import DocumentMatcher
from generic_rag.types import Document, DocumentStatus, FileStorage
from generic_rag.utils.pagination import PaginatedResults, Pagination
from generic_rag.utils.repository import RepositoryMixin

logger = logging.getLogger(__name__)


class DocumentRepository(RepositoryMixin[DocumentEntity]):
    def __init__(self, channel_key: str):
        self._channel_key = channel_key

    async def get_next_id(self) -> int:
        return (
            await get_current_session().scalar(
                select(func.max(DocumentEntity.document_id) + 1).where(
                    DocumentEntity.channel_key == self._channel_key
                )
            )
            or 1
        )

    async def get_by_id(self, id_: int) -> DocumentEntity | None:
        return await get_current_session().scalar(
            select(DocumentEntity).where(
                DocumentEntity.channel_key == self._channel_key,
                DocumentEntity.document_id == id_,
            )
        )

    async def get_by_ids(self, ids: set[int]) -> Sequence[DocumentEntity]:
        result = await get_current_session().scalars(
            select(DocumentEntity).where(
                DocumentEntity.channel_key == self._channel_key, DocumentEntity.document_id.in_(ids)
            )
        )
        return result.all()

    async def get_by_url(self, url: str) -> DocumentEntity | None:
        return await get_current_session().scalar(
            select(DocumentEntity).where(
                DocumentEntity.channel_key == self._channel_key,
                DocumentEntity.url == url,
            )
        )

    async def get_total_count(self, matcher: DocumentMatcher | None) -> int:
        if matcher and (matcher_query := matcher.get_query()) is not None:
            where_clause = and_(
                DocumentEntity.channel_key == self._channel_key,
                DocumentEntity.document_id.in_(matcher_query),
            )
        else:
            where_clause = DocumentEntity.channel_key == self._channel_key

        return (
            await get_current_session().scalar(
                select(func.count()).select_from(DocumentEntity).where(where_clause)
            )
            or 0
        )

    async def list_all(
        self, matcher: DocumentMatcher | None, offset: int, limit: int
    ) -> Sequence[DocumentEntity]:
        if matcher and (matcher_query := matcher.get_query()) is not None:
            where_clause = and_(
                DocumentEntity.channel_key == self._channel_key,
                DocumentEntity.document_id.in_(matcher_query),
            )
        else:
            where_clause = DocumentEntity.channel_key == self._channel_key

        result = await get_current_session().scalars(
            select(DocumentEntity)
            .where(where_clause)
            .order_by(DocumentEntity.document_id.desc())
            .offset(offset)
            .limit(limit)
        )
        return result.all()

    async def set_status(self, document_id: int, status: DocumentStatus):
        await get_current_session().execute(
            update(DocumentEntity)
            .where(
                DocumentEntity.channel_key == self._channel_key,
                DocumentEntity.document_id == document_id,
            )
            .values(status=status)
        )


class _Document(Document):
    content_fetcher: Callable[[], Awaitable[AsyncIterable[bytes] | None]] = Field(
        ..., repr=False, exclude=True
    )

    async def get_content_stream(self) -> AsyncIterable[bytes] | None:
        if stream := await self.content_fetcher():
            return stream
        return None

    @classmethod
    def from_entity(cls, entity: DocumentEntity, file_storage: FileStorage) -> Self:
        content_url = entity.url

        async def content_fetcher() -> AsyncIterable[bytes] | None:
            return await file_storage.download_file(content_url)

        return cls(
            id=entity.document_id,
            display_name=entity.display_name,
            mime_type=entity.mime_type,
            size=entity.size,
            url=entity.url,
            metadata=entity.metadata_ or {},
            content_fetcher=content_fetcher,
            status=entity.status,
        )


@scoped(ScopeName.channel)
class DocumentService:
    """Service for managing documents stored in channel."""

    def __init__(self, channel: Channel, file_storage: FileStorage, chunk_service: ChunkService):
        self._channel = channel
        self._file_storage = file_storage
        self._chunk_service = chunk_service
        self._repository = DocumentRepository(channel.channel_key)

    @transaction
    async def list_documents(
        self, pagination: Pagination, matcher: DocumentMatcher | None = None
    ) -> PaginatedResults[Document]:
        """
        Return list of all documents uploaded to a channel with pagination.

        :param pagination: pagination parameters
        :param matcher: describes required subset of documents
        """
        results = [
            _Document.from_entity(entity, self._file_storage)
            for entity in await self._repository.list_all(
                matcher,
                pagination.offset,
                pagination.limit,
            )
        ]
        total_count = await self._repository.get_total_count(matcher)
        return PaginatedResults.create(results, pagination, total_count)

    async def upload_document(self, folder: str, attachment: UploadFile, metadata: dict | None) -> Document:
        """
        Upload document to a channel.

        :param folder: path of a folder within a channel (can have multiple parts)
        :param attachment: the document to upload
        :param metadata: metadata to assign with document (should match json schema associated with this channel)
        """
        self._validate_attachment(attachment)
        self._validate_metadata(metadata)

        url = await self._upload_attachment(folder, attachment)
        display_name = unquote(os.path.join(folder, attachment.filename))

        return await self._create_document(
            url=url,
            display_name=display_name,
            content_type=attachment.content_type,
            size=attachment.size,
            metadata=metadata,
        )

    @retry(stop=stop_after_attempt(10), retry=retry_if_exception_type(IntegrityError))
    @transaction
    async def _create_document(
        self,
        url: str,
        display_name: str,
        content_type: str,
        size: int,
        metadata: dict | None,
    ) -> Document:
        if (entity := await self._repository.get_by_url(url)) is not None:
            document_id = entity.document_id
            await self._repository.delete(entity)
        else:
            document_id = await self._repository.get_next_id()

        entity = await self._repository.save(
            DocumentEntity(
                channel_key=self._channel.channel_key,
                document_id=document_id,
                status=DocumentStatus.created,
                url=url,
                display_name=display_name,
                mime_type=content_type,
                size=size,
                metadata_=metadata or {},
            )
        )

        return _Document.from_entity(entity, self._file_storage)

    @staticmethod
    def _validate_attachment(attachment: UploadFile):
        if attachment.size < 1:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="Invalid attachment",
            )

        if attachment.content_type != "application/pdf":
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail=f"'{attachment.content_type}': invalid file type",
            )

    def _validate_metadata(self, metadata: dict | None):
        if not metadata:
            return

        try:
            jsonschema.validate(
                metadata,
                self._channel.metadata_schema,
                format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
            )

        except jsonschema.ValidationError as e:
            raise HTTPException(
                status_code=HTTP_422_UNPROCESSABLE_CONTENT, detail=f"Document metadata is not valid: {str(e)}"
            ) from e

    async def _upload_attachment(self, folder: str, attachment: UploadFile) -> str:
        bucket = await self._file_storage.get_bucket()
        basename, ext = os.path.splitext(attachment.filename)
        filename = hashlib.sha1(basename.lower().encode()).hexdigest() + ext
        file_metadata = await self._file_storage.put_file(
            bucket,
            filepath=os.path.join("documents", folder, filename),
            content_type=attachment.content_type,
            content=attachment.file,
        )
        return file_metadata.url

    @transaction
    async def get_document(self, document_id: int) -> Document:
        """
        Get document with given id.

        :param document_id: id of required document
        """
        if document := await self._repository.get_by_id(document_id):
            return _Document.from_entity(document, self._file_storage)

        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    @transaction
    async def get_documents(self, document_ids: Iterable[int]) -> list[Document]:
        """
        Return documents with given IDs.

        :param document_ids: IDs of required documents
        """
        return [
            _Document.from_entity(entity, self._file_storage)
            for entity in await self._repository.get_by_ids(set(document_ids))
        ]

    @transaction
    async def set_document_status(self, document_id: int, status: DocumentStatus):
        """
        Set the processing status for given document.

        :param document_id: id of target document
        :param status: target status
        """
        await self._repository.set_status(document_id, status)

    @transaction
    async def delete_document(self, document_id: int) -> None:
        """
        Delete document with given id and all related data.

        :param document_id: id of required document
        """
        if (document := await self._repository.get_by_id(document_id)) is None:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Document '{document_id}' not found.",
            )

        # cleanup indexes
        tasks = [idx.storage.remove(document.document_id) for idx in await self._channel.get_indexes()]
        await asyncio.gather(*tasks)

        # remove chunks and their data (we cannot rely only on cascade delete
        # because image chunks has files that wouldn't be deleted in that case)
        await self._chunk_service.delete_chunks_by_document(document.document_id)

        # remove the document itself
        await self._file_storage.delete_file(document.url)
        await self._repository.delete(document)
