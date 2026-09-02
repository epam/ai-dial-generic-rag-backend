import asyncio
import itertools
import json
import logging
import os
import zipfile
from collections.abc import AsyncGenerator, Iterable
from io import BytesIO, FileIO
from pathlib import PosixPath
from typing import Annotated, Any, BinaryIO, Literal
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import aiofiles
import anyio
from aidial_sdk.exceptions import InvalidRequestError
from deepdiff import DeepDiff
from fastapi import UploadFile
from injection import scoped
from msgpack import pack, unpack
from opentelemetry.trace import get_tracer
from pydantic import BaseModel, Field, RootModel
from starlette.datastructures import Headers

from generic_rag.channel import Channel
from generic_rag.scope import ScopeName
from generic_rag.services.chunk_service import ChunkService
from generic_rag.services.document_service import DocumentService
from generic_rag.types import AnyChunk, Document, DocumentStatus, FileMetadata, FileStorage, IndexRecord
from generic_rag.utils.iterables import batched_async
from generic_rag.utils.pagination import Pagination
from generic_rag.utils.profile import log_execution_time

tracer = get_tracer(__name__)
logger = logging.getLogger(__name__)


class DocumentRecord(BaseModel):
    """Data of exported document."""

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
    async def create(
        cls, document: Document, chunks: list[AnyChunk], indexes: dict[str, list[IndexRecord[Any]]]
    ):
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


class SerializedChunk(RootModel[Annotated[AnyChunk, Field(discriminator="chunk_type")]]):
    """Utility model used to deserialize chunks."""


@scoped(ScopeName.channel)
class ExportService:
    def __init__(
        self,
        channel: Channel,
        document_service: DocumentService,
        chunk_service: ChunkService,
        file_storage: FileStorage,
    ):
        self._channel = channel
        self._document_service = document_service
        self._chunk_service = chunk_service
        self._file_storage = file_storage

    @log_execution_time(logger)
    async def export_document(self, document: Document, stream: BinaryIO):
        """
        Export given document and its data.

        :param document: the document to export
        :param stream: binary stream to write data
        """
        logger.info(f"exporting document '{document.display_name}'")

        chunks = [chunk async for chunk in self._chunk_service.get_chunks_by_document(document.id)]
        indexes = {
            index.index_name: [record async for record in index.storage.export(document.id)]
            for index in await self._channel.get_indexes()
        }

        document_record = await DocumentRecord.create(document, chunks, indexes)

        return await asyncio.to_thread(pack, document_record.model_dump(), stream)

    @log_execution_time(logger)
    async def export_channel(self, archive_path: str) -> FileMetadata:
        """Export channel's content as single archive uploaded to a file storage."""
        async with anyio.TemporaryDirectory(prefix="export_") as workdir:
            local_path = os.path.join(workdir, PosixPath(archive_path).name)

            with ZipFile(local_path, "w", compression=ZIP_DEFLATED) as zip_file:
                await asyncio.to_thread(
                    zip_file.writestr,
                    "_channel.json",
                    json.dumps(self._channel.dump_config(), indent=2),
                )

                async for batch in batched_async(_iter_documents(self._document_service), 10):
                    await self._export_to_zip_file(batch, zip_file)

            bucket = await self._file_storage.get_bucket()

            with FileIO(local_path, "rb") as stream:
                logger.info(f"uploading archive as '{archive_path}'")
                return await self._file_storage.put_file(
                    bucket,
                    filepath=archive_path,
                    content_type="application/zip",
                    content=stream,
                )

    async def _export_to_zip_file(self, documents: Iterable[Document], zip_file: ZipFile):
        """Export given batch of documents into provided Zip archive."""
        async with anyio.TemporaryDirectory(prefix="batch_") as workdir:

            @tracer.start_as_current_span("export-document")
            async def _export_task(doc: Document) -> str:
                name, _ = os.path.splitext(os.path.basename(doc.display_name))
                with FileIO(
                    os.path.join(
                        workdir,
                        os.path.join(workdir, f"{doc.id}_{name}.msgpack"),
                    ),
                    "wb",
                ) as stream:
                    await self.export_document(doc, stream)
                    return stream.name

            tasks = [asyncio.create_task(_export_task(doc)) for doc in documents]

            for filepath in await asyncio.gather(*tasks):
                logger.info(f"appending '{filepath}' to the archive")
                await asyncio.to_thread(
                    zip_file.write,
                    filepath,
                    arcname=os.path.basename(filepath),
                )

    @log_execution_time(logger)
    async def import_document(self, stream: BinaryIO, overwrite: bool = False) -> Document:
        """
        Import document and its data.

        :param stream: binary stream with serialized data of DocumentRecord
        :param overwrite: allow to overwrite the document that already exists (if any)
        """
        record = DocumentRecord.model_validate(
            await asyncio.to_thread(unpack, stream),
        )

        logger.info(f"importing document '{os.path.join(record.folder, record.filename)}'")

        # import document itself
        document = await self._document_service.create_document(
            attachment=UploadFile(
                BytesIO(record.content_bytes),
                size=len(record.content_bytes),
                filename=record.filename,
                headers=Headers({
                    "content-type": record.mime_type,
                }),
            ),
            folder=record.folder,
            metadata=record.metadata,
            overwrite=overwrite,
        )
        await self._document_service.set_document_status(document.id, record.status)

        # import chunks
        async def _set_document_id(chunks: Iterable[AnyChunk]) -> AsyncGenerator[AnyChunk]:
            for chunk in chunks:
                yield chunk.model_copy(update={"document_id": document.id})

        await self._chunk_service.delete_chunks_by_document(document.id)
        await self._chunk_service.add_chunks(_set_document_id(record.chunks))

        # import indexes
        for index in await self._channel.get_indexes():
            if not (index_data := record.indexes.get(index.index_name)):
                continue

            index_records = [
                record.model_copy(
                    update={"metadata": record.metadata.model_copy(update={"document_id": document.id})}
                )
                for record in index_data
            ]
            await index.storage.remove(document.id)
            await index.storage.add(index_records)

        logger.info(f"successfully imported document '{record.filename}' as '{document.id}'")

        return await self._document_service.get_document(document.id)

    @log_execution_time(logger)
    async def import_channel(self, archive_url: str):
        """
        Import content of given archive into the channel.

        :param archive_url: the URL (within the file storage) of the archive with channel data
        """
        metadata = await self._file_storage.get_file_metadata(archive_url)

        assert metadata is not None
        assert metadata.content_type == "application/zip"

        async with anyio.TemporaryDirectory(prefix="import_") as workdir:
            logger.info(f"downloading channel archive from: {archive_url}")

            async with aiofiles.open(PosixPath(workdir).joinpath(metadata.name), "wb") as fp:
                stream = await self._file_storage.download_file(archive_url)
                assert stream is not None
                async for chunk in stream:
                    await fp.write(chunk)

            with ZipFile(PosixPath(workdir).joinpath(metadata.name)) as zip_file:
                channel_config_name = "_channel.json"
                if not (channel_config_path := zipfile.Path(zip_file, at=channel_config_name)).exists():
                    raise InvalidRequestError(f"The archive does not contain `{channel_config_name}`.")

                channel_config = await asyncio.to_thread(json.load, channel_config_path.open("r"))

                if diff := DeepDiff(
                    self._channel.dump_config(),
                    channel_config,
                    exclude_paths=["channel_key", "retriever", "generation"],
                ):
                    raise InvalidRequestError(
                        "The archive is not compatible with the channel.", detail=json.loads(diff.to_json())
                    )

                for batch in itertools.batched(
                    filter(lambda item: item.filename.endswith(".msgpack"), zip_file.filelist),
                    10,
                    strict=False,
                ):
                    await self._import_from_zip_file(batch, zip_file)

    async def _import_from_zip_file(self, items: Iterable[ZipInfo], zip_file: ZipFile):
        """Import given batch of items from provided zip file into the channel."""
        async with anyio.TemporaryDirectory(prefix="batch_") as workdir:

            @tracer.start_as_current_span("import-document")
            async def _import_task(source_path: str):
                with FileIO(source_path, "rb") as stream:
                    await self.import_document(stream, overwrite=True)

            extracted_files = []

            for zip_info in items:
                logger.info(f"extracting '{zip_info.filename}'")
                extracted_files.append(
                    await asyncio.to_thread(zip_file.extract, zip_info, workdir),
                )

            tasks = [_import_task(item) for item in extracted_files]
            await asyncio.gather(*tasks)


async def _iter_documents(document_service: DocumentService) -> AsyncGenerator[Document]:
    pagination = Pagination(offset=0, limit=25)
    while True:
        result = await document_service.list_documents(pagination)
        if result.results:
            for doc in result.results:
                yield doc
            pagination = Pagination(
                offset=pagination.offset + len(result.results),
                limit=pagination.limit,
            )
        else:
            break
