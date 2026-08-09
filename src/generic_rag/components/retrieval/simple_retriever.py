from collections.abc import Sequence

from injection import inject

from generic_rag.channel import Channel
from generic_rag.components.retrieval.abstract_retriever import AbstractRetriever, AbstractRetrieverConfig
from generic_rag.types import RetrievedDocument
from generic_rag.utils.ranking import rank_fusion


class SimpleRetriever[ConfigT: AbstractRetrieverConfig = AbstractRetrieverConfig](AbstractRetriever[ConfigT]):
    """Retriever that combines result from indexes using rank fusion with equal weighting for all indexes."""

    @inject
    def __init__(self, config: ConfigT, channel: Channel):
        super().__init__(config, channel)

    async def _get_combined_results(
        self, doc_lists: list[Sequence[RetrievedDocument]]
    ) -> Sequence[RetrievedDocument]:
        return await rank_fusion(
            doc_lists,
            key=lambda doc: doc.chunks[0].get_identity(),
        )
