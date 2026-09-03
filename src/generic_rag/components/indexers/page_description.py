import asyncio
import io
import itertools
import logging
from asyncio import Semaphore, TaskGroup
from collections.abc import Collection, Iterable
from typing import Any, Literal

from datauri import DataURI
from injection import inject
from langchain_core.messages import HumanMessage
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

from generic_rag.types import (
    AnyChunk,
    ImageChunk,
    ImageType,
    Indexer,
    IndexRecord,
    IndexRecordMeta,
    LlmConfig,
    ModelProvider,
    VectorType,
)
from generic_rag.utils.profile import log_execution_time

logger = logging.getLogger(__name__)

PAGE_DESCRIPTION_PROMPT_TEMPLATE = """
Please create a detailed description of the provided page image for a search index.
Ignore page header, footer, basic logo and background.
Text with bullet points is NOT a table or image.

Describe every table and every chart on the page. These descriptions are searched by users
asking about specific values, so name the concrete things a query could mention:

- the title or caption, and the subject of the table or chart
- the units and currencies used, and the years or time period covered
- for a table: the row and column names, and the values of notable cells — totals,
  subtotals, extremes, and any figures highlighted by the document
- for a chart: the axis labels, the legend series names, and the approximate values of
  notable points — start, end, peaks, and crossovers, each with its unit
- the countries, regions, companies and perils involved

Describe other images (photo, illustration, diagram) briefly.

Use only information visible on the page.
DO NOT make up an answer.

Make sure to properly escape special characters, like double quotes, in string fields.
"""

PAGE_DESCRIPTION_DEFAULT_LLM_DEPLOYMENT = "gpt-4.1-mini-2025-04-14"
# matches the default page image size of the page extractor, so descriptions are generated
# from the same resolution the page was stored at instead of a downscaled copy
PAGE_DESCRIPTION_MAX_IMAGE_SIZE = 1536

# Error message in the openai library tells to use math.inf, but the type for the max_retries is int
MAX_RETRIES = 1_000_000_000  # One billion retries should be enough


type DescriptionKind = Literal[
    "page_summary",
    "page_key_fact",
    "image_summary",
    "image_key_fact",
    "table_summary",
    "table_key_fact",
]


class DescriptionItem(BaseModel):
    text: str
    kind: DescriptionKind
    element_index: int | None = None


class ImageDescription(BaseModel):
    """Image description"""

    image_summary: str = Field(description="the summary of the image description")
    key_fact: str = Field(description="the most important fact from the image")

    model_config = ConfigDict(
        hide_input_in_errors=True,
        extra="forbid",
    )


class TableDescription(BaseModel):
    """Table description"""

    table_summary: str = Field(description="the summary of the table description")
    key_fact: str = Field(description="the most important fact from the table")

    model_config = ConfigDict(
        hide_input_in_errors=True,
        extra="forbid",
    )


class PageDescription(BaseModel):
    """Page description"""

    page_summary: str = Field(description="the summary of the page description")
    key_fact: str = Field(description="the most important fact from the page")
    images: list[ImageDescription] = Field(
        description="the array of the descriptions for the images on the page",
        default_factory=list,
    )
    tables: list[TableDescription] = Field(
        description="the array of the descriptions for the tables on the page",
        default_factory=list,
    )

    model_config = ConfigDict(
        hide_input_in_errors=True,
        extra="forbid",
    )

    def flatten(self) -> Collection[DescriptionItem]:
        result: list[DescriptionItem] = []

        def _add_to_result(text: str, kind: DescriptionKind, element_index: int | None):
            if text := text.replace("\n", " ").replace("\r", " ").replace("\u0000", "").strip():
                result.append(
                    DescriptionItem(
                        text=text,
                        kind=kind,
                        element_index=element_index,
                    )
                )

        _add_to_result(self.page_summary, "page_summary", None)
        _add_to_result(self.key_fact, "page_key_fact", None)

        for i, image in enumerate(self.images):
            _add_to_result(image.image_summary, "image_summary", i)
            _add_to_result(image.key_fact, "image_key_fact", i)

        for i, table in enumerate(self.tables):
            _add_to_result(table.table_summary, "table_summary", i)
            _add_to_result(table.key_fact, "table_key_fact", i)

        return result


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


class PageDescriptionConfig(BaseModel):
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
    embeddings: TextEmbeddingsConfig = Field(
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


class PageDescriptionIndexer(Indexer[VectorType, PageDescriptionConfig]):
    """Uses vision model to generate descriptions of page images to enable search on them."""

    @inject
    def __init__(self, config: PageDescriptionConfig, model_provider: ModelProvider):
        super().__init__(config)

        self._llm = model_provider.get_llm(config.llm)
        self._embeddings = model_provider.get_embeddings_model(
            deployment=config.embeddings.deployment_name,
            max_retries=config.embeddings.max_retries,
        )

    async def index_query(self, query: str) -> VectorType:
        return await self._embeddings.aembed_query(query)

    @log_execution_time(logger)
    async def index_data(
        self, data: Iterable[tuple[AnyChunk | str, IndexRecordMeta]]
    ) -> Collection[IndexRecord[VectorType]]:
        semaphore = Semaphore(self.config.max_concurrency)

        async def _get_page_description_task(
            chunk: ImageChunk, meta: IndexRecordMeta
        ) -> Collection[tuple[str, IndexRecordMeta]]:
            async with semaphore:
                page_description = await self._get_page_description(chunk)
                return [
                    (
                        item.text,
                        meta.model_copy(
                            update=item.model_dump(exclude_none=True),
                        ),
                    )
                    for item in page_description.flatten()
                ]

        async with TaskGroup() as task_group:
            tasks = [
                task_group.create_task(_get_page_description_task(chunk, meta))
                for chunk, meta in data
                if isinstance(chunk, ImageChunk) and chunk.image_type == ImageType.page
            ]

        texts: list[str] = []
        record_metas: list[IndexRecordMeta] = []

        for description, record_meta in itertools.chain(*[task.result() for task in tasks]):
            texts.append(description)
            record_metas.append(record_meta)

        embeddings = await self._embeddings.aembed_documents(texts)

        return [
            IndexRecord(index=index, metadata=meta)
            for index, meta in zip(embeddings, record_metas, strict=True)
        ]

    @log_execution_time(logger)
    async def _get_page_description(self, chunk: ImageChunk) -> PageDescription:
        assert chunk.image_type == ImageType.page

        prompt = await asyncio.to_thread(self._build_prompt, chunk)
        llm_chain = self._llm.with_structured_output(
            PageDescription.model_json_schema(), method="json_schema", strict=True
        )
        response = await llm_chain.ainvoke([HumanMessage(prompt)])
        return PageDescription.model_validate(response)

    def _build_prompt(self, chunk: ImageChunk, image_details: Literal["low", "high", "auto"] = "auto"):
        image = Image.open(io.BytesIO(chunk.content))
        image_size = max(image.width, image.height)

        if image_size > self.config.max_image_size:
            if image.width > image.height:
                target_width = self.config.max_image_size
                target_height = round(image.height * (self.config.max_image_size / image.width))
            else:
                target_width = round(image.width * (self.config.max_image_size / image.height))
                target_height = self.config.max_image_size

            image = image.resize(size=(target_width, target_height))

            with io.BytesIO() as fp:
                image.save(fp, format="png")
                image_uri = DataURI.make(
                    "image/png",
                    charset=None,
                    base64=True,
                    data=fp.getvalue(),
                )
        else:
            image_uri = chunk.get_data_uri()

        return [
            {"type": "text", "text": PAGE_DESCRIPTION_PROMPT_TEMPLATE},
            {
                "type": "image_url",
                "image_url": {
                    "url": str(image_uri),
                    "detail": image_details,
                },
            },
        ]
