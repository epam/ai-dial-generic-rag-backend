import hashlib
import logging
import os
from collections.abc import AsyncIterable, Awaitable, Callable, Iterable, Sequence
from pathlib import PosixPath
from typing import Self
from urllib.parse import unquote

import jsonschema
from aidial_sdk.exceptions import InvalidRequestError, RequestValidationError, ResourceNotFoundError
from fastapi import UploadFile
from injection import scoped
from pydantic import Field
from sqlalchemy import and_, func, select, update
from sqlalchemy.exc import IntegrityError
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from generic_rag.channel import Channel
from generic_rag.db.entities import DocumentEntity
from generic_rag.db.session import get_current_session, transaction
from generic_rag.scope import ScopeName
from generic_rag.services.document_matcher import DocumentMatcher
from generic_rag.types import Document, DocumentStatus, FileMetadata, FileStorage
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

    async def exists(self, document_id: int) -> bool:
        return (
            await get_current_session().scalar(
                select(DocumentEntity.document_id)
                .where(
                    DocumentEntity.channel_key == self._channel_key,
                    DocumentEntity.document_id == document_id,
                )
                .exists()
                .select()
            )
            or False
        )

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

    def __init__(self, channel: Channel, file_storage: FileStorage):
        self._channel = channel
        self._file_storage = file_storage
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

    @transaction
    async def create_document(
        self, attachment: UploadFile, folder: str | None = None, metadata: dict | None = None
    ) -> Document:
        """
        Upload document to a channel.

        :param attachment: the file to upload
        :param folder: path of a target folder within a channel (can have multiple parts)
        :param metadata: metadata to assign with document (should match JSON schema associated with this channel)
        """
        assert attachment.filename
        assert attachment.content_type

        self._validate_attachment(attachment)
        self._validate_metadata(metadata)

        bucket = await self._file_storage.get_bucket()
        upload_path = self._get_upload_filepath(attachment.filename, folder)

        file_meta = await self._file_storage.put_file(
            bucket,
            filepath=upload_path,
            content_type=attachment.content_type,
            content=attachment.file,
        )

        display_name = self._get_display_name(file_meta.url, attachment.filename)

        if (entity := await self._repository.get_by_url(file_meta.url)) is not None:
            if entity.etag != file_meta.etag:
                entity.status = DocumentStatus.created

            entity.etag = file_meta.etag
            entity.mime_type = file_meta.content_type
            entity.size = file_meta.content_length

            entity.display_name = display_name
            entity.metadata_ = metadata or {}

            return _Document.from_entity(await self._repository.save(entity), self._file_storage)

        return await self._create_document(file_meta, display_name, metadata)

    @transaction
    async def update_document(
        self, document_id: int, attachment: UploadFile | None = None, metadata: dict | None = None
    ) -> Document:
        """
        Update document with given ID by replacing its content and/or metadata.

        :param document_id: the id of required document
        :param attachment: the file to replace the document's content with
        :param metadata: metadata to assign with document (should match JSON schema associated with this channel)
        """
        if (entity := await self._repository.get_by_id(document_id)) is None:
            raise ResourceNotFoundError(f"Document '{document_id}' not found.")

        if attachment is not None:
            assert attachment.content_type

            self._validate_attachment(attachment)

            bucket = await self._file_storage.get_bucket()
            upload_path = str(PosixPath(entity.url).relative_to(PosixPath(f"files/{bucket}")))

            file_meta = await self._file_storage.put_file(
                bucket,
                filepath=upload_path,
                content_type=attachment.content_type,
                content=attachment.file,
            )
            assert file_meta.url == entity.url

            if entity.etag != file_meta.etag:
                entity.status = DocumentStatus.created

            entity.etag = file_meta.etag
            entity.mime_type = file_meta.content_type
            entity.size = file_meta.content_length

        if metadata is not None:
            self._validate_metadata(metadata)
            entity.metadata_ = metadata

        return _Document.from_entity(await self._repository.save(entity), self._file_storage)

    @retry(stop=stop_after_attempt(10), retry=retry_if_exception_type(IntegrityError))
    @transaction
    async def _create_document(
        self, file_meta: FileMetadata, display_name: str, metadata: dict | None
    ) -> _Document:
        return _Document.from_entity(
            await self._repository.save(
                DocumentEntity(
                    channel_key=self._channel.channel_key,
                    document_id=await self._repository.get_next_id(),
                    status=DocumentStatus.created,
                    url=file_meta.url,
                    etag=file_meta.etag,
                    display_name=display_name,
                    mime_type=file_meta.content_type,
                    size=file_meta.content_length,
                    metadata_=metadata or {},
                )
            ),
            self._file_storage,
        )

    @staticmethod
    def _validate_attachment(attachment: UploadFile):
        if not (attachment.size and attachment.content_type and attachment.filename):
            raise InvalidRequestError("Invalid attachment")
        if attachment.content_type != "application/pdf":
            raise InvalidRequestError(f"'{attachment.content_type}': unsupported file type")

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
            raise RequestValidationError(
                str(e), display_message=f"Value of metadata violates JSON schema: {e.message}"
            ) from e

    @staticmethod
    def _get_upload_filepath(filename: str, folder: str | None = None) -> str:
        """
        Return full path with an application bucket for uploading file with given filename and folder.

        :param filename: the filename of source file
        :param folder: target folder where the file should be uploaded
        """
        basename, ext = os.path.splitext(filename.strip())
        target_folder = PosixPath((folder or "").strip().lstrip("/"))
        target_name = hashlib.sha1(basename.lower().encode()).hexdigest() + ext
        return str(PosixPath("documents", *target_folder.parts, target_name))

    @staticmethod
    def _get_display_name(url: str, original_filename: str):
        file_path = PosixPath(unquote(url))
        return str(file_path.relative_to(PosixPath(*file_path.parts[:3])).with_name(original_filename))

    @transaction
    async def get_document(self, document_id: int) -> Document:
        """
        Get document with given id.

        :param document_id: id of required document
        """
        if document := await self._repository.get_by_id(document_id):
            return _Document.from_entity(document, self._file_storage)

        raise ResourceNotFoundError(f"Document '{document_id}' not found.")

    @transaction
    async def get_documents_by_id(self, document_ids: Iterable[int]) -> list[Document]:
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
    async def exists(self, document_id) -> bool:
        """
        Check if the document with given ID exists.

        :param document_id: id of required document
        """
        return await self._repository.exists(document_id)

    @transaction
    async def delete_document(self, document_id: int) -> None:
        """
        Delete document with given id.

        :param document_id: id of required document
        """
        if (entity := await self._repository.get_by_id(document_id)) is None:
            raise ResourceNotFoundError(f"Document '{document_id}' not found.")

        await self._file_storage.delete_file(entity.url)
        await self._repository.delete(entity)
