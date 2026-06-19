import logging
import os
from collections.abc import AsyncGenerator, Iterable
from io import BytesIO
from typing import Annotated, Any, Literal

from fastapi import UploadFile
from injection import scoped
from msgpack import packb, unpackb
from pydantic import BaseModel, Field, RootModel
from starlette.datastructures import Headers

from generic_rag.channel import Channel
from generic_rag.scope import ScopeName
from generic_rag.services.chunk_service import ChunkService
from generic_rag.services.document_service import DocumentService
from generic_rag.types import AnyChunk, Document, DocumentStatus, IndexRecord
from generic_rag.utils.profile import log_execution_time

logger = logging.getLogger(__name__)


class DocumentRecord(BaseModel):
    """ Data of exported document. """
    version: Literal["v2"] = "v2"
    filename: str
    folder: str
    mime_type: str
    metadata: dict | None
    status: DocumentStatus
    content_bytes: bytes
    chunks: list[AnyChunk]
    indexes: dict[str, list[IndexRecord[Any]]]

    @classmethod
    async def create(cls, document: Document, chunks: list[AnyChunk], indexes: dict[str, list[IndexRecord[Any]]]):
        folder, filename = os.path.split(document.display_name)
        return cls(
            filename=filename,
            folder=folder,
            mime_type=document.mime_type,
            metadata=document.metadata,
            status=document.status,
            content_bytes=await document.get_content(),
            chunks=chunks,
            indexes=indexes,
        )


class SerializedChunk(
    RootModel[
        Annotated[
            AnyChunk,
            Field(discriminator="chunk_type")
        ]
    ]
):
    """ Utility model used to deserialize chunks. """


@scoped(ScopeName.channel)
class ExportService:
    def __init__(self, channel: Channel, document_service: DocumentService, chunk_service: ChunkService):
        self._channel = channel
        self._document_service = document_service
        self._chunk_service = chunk_service

    @log_execution_time(logger)
    async def export_document(self, document: Document) -> bytes:
        """ Export given document and its related data. """
        logger.info(f"exporting document '{document.display_name}'")

        chunks = [chunk async for chunk in self._chunk_service.get_chunks_by_document(document.id)]
        indexes = {
            index.index_name: [
                record async for record in index.storage.export(document.id)
            ]
            for index in await self._channel.get_indexes()
        }

        document_record = await DocumentRecord.create(document, chunks, indexes)

        return packb(
            document_record.model_dump()
        )

    @log_execution_time(logger)
    async def import_document(self, document_data: bytes) -> Document:
        """ Import document and its related data. """
        record = DocumentRecord.model_validate(
            unpackb(document_data)
        )

        logger.info(f"importing document '{os.path.join(record.folder, record.filename)}'")

        # import document itself
        document = await self._document_service.upload_document(
            folder=record.folder,
            attachment=UploadFile(
                BytesIO(record.content_bytes),
                size=len(record.content_bytes),
                filename=record.filename,
                headers=Headers({
                    "content-type": record.mime_type,
                }),
            ),
            metadata=record.metadata,
        )
        await self._document_service.set_document_status(document.id, record.status)

        # import chunks
        async def _set_document_id(chunks: Iterable[AnyChunk]) -> AsyncGenerator[AnyChunk]:
            for chunk in chunks:
                yield chunk.model_copy(
                    update={
                        "document_id": document.id
                    }
                )

        await self._chunk_service.delete_chunks_by_document(document.id)
        await self._chunk_service.add_chunks(
            _set_document_id(record.chunks)
        )

        # import indexes
        for index in await self._channel.get_indexes():
            if not (index_data := record.indexes.get(index.index_name)):
                continue

            index_records = [
                record.model_copy(
                    update={
                        "metadata": record.metadata.model_copy(
                            update={
                                "document_id": document.id
                            }
                        )
                    }
                )
                for record in index_data
            ]
            await index.storage.remove(document.id)
            await index.storage.add(index_records)

        logger.info(
            f"successfully imported document '{record.filename}' as '{document.id}'"
        )

        return await self._document_service.get_document(document.id)
