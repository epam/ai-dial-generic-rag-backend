from injection import inject
from langchain.retrievers import EnsembleRetriever

from generic_rag.channel import Channel
from generic_rag.components.retrieval.abstract_retriever import AbstractRetriever, AbstractRetrieverConfig
from generic_rag.components.retrieval.index_retriever import IndexRetriever
from generic_rag.components.retrieval.retrieval_stage import RetrievalStage
from generic_rag.components.search_index import ChunkIndex
from generic_rag.services.chunk_service import ChunkService


class SimpleRetriever[ConfigT: AbstractRetrieverConfig = AbstractRetrieverConfig](AbstractRetriever[ConfigT]):
    """ Retriever that combines result from indexes using rank fusion with equal weighting for all indexes. """

    @inject
    def __init__(self, config: ConfigT, channel: Channel, chunk_service: ChunkService):
        super().__init__(config, channel)
        self._chunk_service = chunk_service

    def _create_final_stage(self, intermediate_stages: list[RetrievalStage]) -> RetrievalStage:
        if len(intermediate_stages) == 1:
            return intermediate_stages[0]

        return self._stage_factory(
            "Combined search (simple)",
            EnsembleRetriever(
                retrievers=intermediate_stages,
                weights=[1.0] * len(intermediate_stages),
                id_key="identity",
            )
        )

    def _create_intermediate_stage(self, index: ChunkIndex, documents: list[int] | None, top_k: int) -> RetrievalStage:
        """ Create :class:`RetrievalStage` for given index. """
        return self._stage_factory(
            index.display_name,
            IndexRetriever(
                index=index,
                documents=documents,
                top_k=top_k,
                chunk_service=self._chunk_service,
            )
        )
