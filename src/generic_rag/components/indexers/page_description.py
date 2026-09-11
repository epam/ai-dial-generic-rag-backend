import logging
from collections.abc import Collection, Iterable
from typing import Any

import jsonpath_ng
from pydantic import BaseModel, Field, field_validator

from generic_rag.components.indexers.text_embeddings import TextEmbeddingsConfig, TextEmbeddingsIndexer
from generic_rag.components.transform.page_description import (
    MAX_RETRIES,
    PAGE_DESCRIPTION_DEFAULT_LLM_DEPLOYMENT,
    PAGE_DESCRIPTION_MAX_IMAGE_SIZE,
    PageDescriptionConfig,
)
from generic_rag.types import (
    AnyChunk,
    ChunkIndexer,
    ChunkRef,
    ChunkTransform,
    ImageChunk,
    ImageType,
    IndexRecord,
    IndexRecordMeta,
    LlmConfig,
    VectorType,
)
from generic_rag.utils.iterables import iterate_async
from generic_rag.utils.profile import log_execution_time

logger = logging.getLogger(__name__)


class EmbeddingsConfig(BaseModel):
    deployment_name: str = Field(
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


class PageDescriptionIndexerConfig(BaseModel):
    llm: LlmConfig = Field(
        default=LlmConfig(
            deployment_name=PAGE_DESCRIPTION_DEFAULT_LLM_DEPLOYMENT,
            max_retries=MAX_RETRIES,
        ),
        description=(
            "Configuration for the LLM used in the description index. "
            "The model should support vision. "
            "The model will be used for every page of the document, so "
            "cheap and fast models are preferred."
        ),
    )
    embeddings: EmbeddingsConfig = Field(
        description="Configuration of embeddings model used for indexing of pages descriptions."
    )
    max_image_size: int = Field(
        PAGE_DESCRIPTION_MAX_IMAGE_SIZE,
        description=(
            "Maximum size of page image to be sent into LLM. "
            "If the image chunk is bigger, it will be resized to fit that value."
        ),
    )
    max_concurrency: int = Field(
        default=2,
        description="Maximum number of concurrent requests sent to LLM",
    )

    @field_validator("llm", mode="before")
    @classmethod
    def merge_llm_defaults(cls, data: Any):
        if isinstance(data, dict):
            default_value = cls.model_fields["llm"].default.model_dump()
            return default_value | data
        return data


class PageDescriptionIndexer(ChunkIndexer[VectorType, PageDescriptionIndexerConfig]):
    """
    Uses vision model to generate descriptions of page images to enable search on them.

    **DEPRECATED** and will be removed in future versions.

    Internally it uses composition of `page_description` transform and `text_embedding` indexer.
    You should define them explicitly in your pipeline as illustrated below instead of using this component:

    ```json
    {
      "parsers": [
        {
          "type": "page_extractor",
          "transform": [
            {
              "type": "page_description",
              "llm": {
                "deployment_name": "gpt-5.6-luna-2026-07-09"
              }
            }
          ]
        }
      ],
      "indexes": {
        "page-description-embedding": {
          "type": "chunk",
          "display_name": "Page image descriptions search",
          "indexer": {
            "type": "text_embeddings",
            "target": "$.description[*].text",
            "deployment_name": "text-embedding-3-large"
          }
        }
      }
    }
    ```
    """

    def __init__(self, config: PageDescriptionIndexerConfig):
        super().__init__(config)

        self._transform = ChunkTransform.create(
            PageDescriptionConfig.model_validate(
                dict(
                    field="description",
                    **self.config.model_dump(),
                ),
            ),
        )
        self._indexer = TextEmbeddingsIndexer(
            TextEmbeddingsConfig.model_validate(
                self.config.embeddings.model_dump(),
            )
        )
        self._description_path = jsonpath_ng.parse("$.description[*]")

    async def index_query(self, query: str) -> VectorType:
        return await self._indexer.index_query(query)

    @log_execution_time(logger)
    async def index_chunks(
        self, data: Iterable[tuple[AnyChunk, IndexRecordMeta]]
    ) -> Collection[IndexRecord[VectorType]]:
        chunks: list[ImageChunk] = []
        meta_by_ref: dict[ChunkRef, IndexRecordMeta] = {}

        for chunk, meta in data:
            if not (isinstance(chunk, ImageChunk) and chunk.image_type == ImageType.page):
                continue
            chunks.append(chunk)
            meta_by_ref[chunk.get_identity()] = meta

        transformed_data = [
            (
                match.value["text"],
                meta_by_ref[chunk.get_identity()].model_copy(update=match.value),
            )
            async for chunk in self._transform.apply(iterate_async(chunks))
            for match in self._description_path.find(chunk.metadata.model_dump())
            if isinstance(match.value, dict) and isinstance(match.value.get("text"), str)
        ]

        return await self._indexer.index_strings(transformed_data)
