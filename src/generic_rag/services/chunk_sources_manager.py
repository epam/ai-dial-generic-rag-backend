import asyncio
from collections.abc import Generator
from typing import Self, cast

from injection import scoped
from langchain_core.documents import Document as LangchainDocument
from pydantic import BaseModel, Field

from generic_rag.scope import ScopeName
from generic_rag.services.document_service import DocumentService
from generic_rag.types import AnyChunk, FileStorage


class ChunkSource(BaseModel):
    """Information about the source of the chunk."""

    source_url: str = Field(
        ...,
        description="url of the chunk source",
    )
    source_display_name: str = Field(
        ...,
        description="name of the chunk source that can be displayed to user",
    )
    source_metadata: dict = Field(
        default_factory=dict,
        description="metadata associated with chunk source",
    )

    @classmethod
    def from_chunk(cls, chunk: AnyChunk) -> Self:
        return cls.model_validate(chunk.model_extra)


@scoped(ScopeName.channel)
class ChunkSourcesManager:
    """
    Utility class for managing chunk sources:

    * adds information about source document to every loaded chunk
    * shares source document with a user that performs request
      (by copying the source document to user's bucket)
    """

    def __init__(self, document_service: DocumentService, file_storage: FileStorage):
        self._lock = asyncio.Lock()
        self._document_service = document_service
        self._file_storage = file_storage
        self._sources: dict[int, ChunkSource] = {}
        """ cache of previously loaded sources """

    async def add_sources(self, retrieved_docs: list[LangchainDocument]):
        """Add fields of :class:`ChunkSource` to chunks of given retrieved documents."""
        async with self._lock:
            await self._load_sources(retrieved_docs)

        for doc in retrieved_docs:
            doc.metadata["chunks"] = [
                chunk.model_copy(update=self._sources[chunk.document_id].model_dump())
                for chunk in self._iter_chunks(doc)
            ]

    async def _load_sources(self, retrieved_docs: list[LangchainDocument]):
        if not (
            missing_sources := {
                chunk.document_id
                for doc in retrieved_docs
                for chunk in self._iter_chunks(doc)
                if chunk.document_id not in self._sources
            }
        ):
            return

        for document in await self._document_service.get_document_list(missing_sources):
            source_url = await self._file_storage.copy_file_to_user(
                document.url,
                document.display_name,
            )
            self._sources[document.id] = ChunkSource(
                source_url=source_url,
                source_display_name=document.display_name,
                source_metadata=document.metadata,
            )

    @staticmethod
    def _iter_chunks(doc: LangchainDocument) -> Generator[AnyChunk]:
        yield from cast(list[AnyChunk], doc.metadata.get("chunks", []))
