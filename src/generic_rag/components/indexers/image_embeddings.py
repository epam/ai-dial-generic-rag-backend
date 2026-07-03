import asyncio
import base64
import logging
from asyncio import Semaphore
from collections.abc import Collection, Iterable
from functools import cache

from injection import inject
from pydantic import BaseModel, Field

from generic_rag.types import (
    AnyChunk,
    ImageChunk,
    IndexedEntityMeta,
    Indexer,
    IndexRecord,
    ModelProvider,
    VectorType,
)
from generic_rag.utils.profile import log_execution_time

logger = logging.getLogger(__name__)


class ImageEmbeddingsConfig(BaseModel):
    deployment_name: str = Field(
        ...,
        description="Name of a model deployment to use.",
        examples=[
            "multimodalembedding@001",
            "azure-ai-vision-embeddings",
        ],
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries to make when performing requests to the model.",
    )


class ImageEmbeddingsIndexer(Indexer[VectorType, ImageEmbeddingsConfig]):
    """Represent source images as vectors calculated with multimodal embeddings model."""

    @staticmethod
    @cache
    def _get_model_semaphore():
        return Semaphore(10)  # todo: get from config

    @inject
    def __init__(self, config: ImageEmbeddingsConfig, model_provider: ModelProvider):
        super().__init__(config)

        self._model = model_provider.get_embeddings_model(
            deployment=self.config.deployment_name,
            max_retries=self.config.max_retries,
        )
        self._semaphore = self._get_model_semaphore()

    async def index_query(self, query: str) -> VectorType:
        return await self._model.aembed_query(query)

    @log_execution_time(logger)
    async def index_data(
        self, data: Iterable[tuple[AnyChunk | str, IndexedEntityMeta]]
    ) -> Collection[IndexRecord[VectorType]]:
        async def _embed_image_task(chunk: ImageChunk, meta: IndexedEntityMeta):
            async with self._semaphore:
                return await self._embed_image(chunk, meta)

        tasks = [_embed_image_task(item, meta) for item, meta in data if isinstance(item, ImageChunk)]

        return await asyncio.gather(*tasks)

    async def _embed_image(self, chunk: ImageChunk, meta: IndexedEntityMeta) -> IndexRecord[VectorType]:
        image_base64 = base64.b64encode(chunk.content).decode("utf-8")
        response = await self._model.async_client.create(
            model=self._model.deployment,
            input=[],
            extra_body={
                "custom_input": [
                    {
                        "type": chunk.mime_type,
                        "data": image_base64,
                    }
                ],
            },
            encoding_format="float",
        )
        assert len(response.data) == 1
        return IndexRecord(index=response.data[0].embedding, metadata=meta)
