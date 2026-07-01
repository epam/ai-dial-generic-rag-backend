from langchain_core.documents import Document as LangchainDocument
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from generic_rag.components.search_index import ChunkIndex
from generic_rag.services.chunk_service import ChunkService


class IndexRetriever(BaseRetriever):
    """
    Retriever that returns chunks found by given index as separate langchain documents.
    """

    index: ChunkIndex = Field(repr=False)
    documents: list[int] | None = None
    top_k: int
    chunk_service: ChunkService = Field(repr=False)

    def _get_relevant_documents(self, query: str, *args, **kwargs) -> list[LangchainDocument]:
        raise NotImplementedError()

    async def _aget_relevant_documents(self, query: str, *args, **kwargs) -> list[LangchainDocument]:
        chunk_refs = await self.index.search(
            query,
            limit=self.top_k,
            documents=self.documents,
        )
        chunks = {
            chunk.get_identity(): chunk
            for chunk in await self.chunk_service.get_chunks_by_references(chunk_refs)
        }
        return [
            LangchainDocument(
                page_content="",
                metadata={
                    "identity": chunk.get_identity(),
                    "chunks": [chunk],
                },
            )
            for chunk_ref in chunk_refs
            if (chunk := chunks.get(chunk_ref)) is not None
        ]
