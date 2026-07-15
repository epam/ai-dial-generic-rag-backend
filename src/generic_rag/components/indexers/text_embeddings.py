import logging
from collections.abc import Collection

from injection import inject
from pydantic import BaseModel, Field

from generic_rag.components.indexers.text_indexer import TextIndexer
from generic_rag.types import IndexedEntityMeta, IndexRecord, ModelProvider, VectorType

logger = logging.getLogger(__name__)


class TextEmbeddingsConfig(BaseModel):
    deployment_name: str = Field(
        ...,
        description="Name of a text embeddings model to use.",
        examples=[
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ],
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries to make when performing requests to the model.",
    )


class TextEmbeddingsIndexer(TextIndexer[VectorType, TextEmbeddingsConfig]):
    """Represent source texts as vectors calculated with text embeddings model."""

    @inject
    def __init__(self, config: TextEmbeddingsConfig, model_provider: ModelProvider):
        super().__init__(config)

        self._model = model_provider.get_embeddings_model(
            deployment=self.config.deployment_name,
            max_retries=self.config.max_retries,
        )

    async def index_query(self, query: str) -> VectorType:
        return await self._model.aembed_query(query)

    async def _index_texts(
        self, texts: list[str], record_metas: list[IndexedEntityMeta]
    ) -> Collection[IndexRecord[VectorType]]:
        embeddings = await self._model.aembed_documents(texts)

        return [
            IndexRecord(index=index, metadata=meta)
            for index, meta in zip(embeddings, record_metas, strict=True)
        ]
