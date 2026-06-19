import asyncio
import logging

from langchain_core.documents import Document as LangchainDocument
from langchain_core.retrievers import BaseRetriever
from opentelemetry.trace import get_tracer
from pydantic import Field

from generic_rag.services.chunk_sources_manager import ChunkSourcesManager
from generic_rag.types import RetrievalStageListener

tracer = get_tracer(__name__)
logger = logging.getLogger(__name__)


class FailedRetrieverError(Exception):
    ...


class RetrievalStage(BaseRetriever):
    """ Represents dedicated stage of retrieval process. """
    stage_name: str
    retriever: BaseRetriever = Field(repr=False)
    listener: RetrievalStageListener = Field(repr=False)
    sources_manager: ChunkSourcesManager = Field(repr=False)

    def _get_relevant_documents(self, query: str, *args, **kwargs) -> list[LangchainDocument]:
        return asyncio.run(self._aget_relevant_documents(query))

    @tracer.start_as_current_span("retriever-stage")
    async def _aget_relevant_documents(self, query: str, *args, **kwargs) -> list[LangchainDocument]:
        try:
            async with self.listener.begin(self.stage_name):
                logger.info(f"Running stage '{self.stage_name}'")
                try:
                    retrieved_docs = await self.retriever.ainvoke(query)
                except Exception as e:
                    await self.listener.on_error(e)
                    raise FailedRetrieverError(str(e)) from e

                logger.info(f"'{self.stage_name}': found {len(retrieved_docs)} reference(s)")

                await self.sources_manager.add_sources(retrieved_docs)
                await self.listener.on_retrieval_result(retrieved_docs)

            return retrieved_docs

        except FailedRetrieverError as e:
            logger.warning(f"Internal error in '{self.stage_name}': {str(e)}", exc_info=e)
            return []
