import asyncio
import io
import logging
from asyncio import Semaphore, TaskGroup
from collections.abc import AsyncIterable, Collection
from typing import Any, Literal

from datauri import DataURI
from injection import inject
from langchain_core.messages import HumanMessage
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from generic_rag.types import AnyChunk, ChunkTransform, ImageChunk, ImageType, LlmConfig, ModelProvider
from generic_rag.utils.iterables import batched_async
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
PAGE_DESCRIPTION_MAX_IMAGE_SIZE = 800

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


class PageDescriptionConfig(BaseModel):
    field: str = Field(
        default="description",
        description="The field within chunk's metadata to save generated descriptions",
    )
    llm: LlmConfig = Field(
        default=LlmConfig(
            deployment_name=PAGE_DESCRIPTION_DEFAULT_LLM_DEPLOYMENT,
            max_retries=MAX_RETRIES,
        ),
        description=(
            "Configuration for the LLM used in the description index. "
            "The model should support vision. "
            "The model will be used for every image chunk with page of the document, "
            "so cheap and fast models are preferred."
        ),
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


class PageDescriptionChunkTransform(ChunkTransform[PageDescriptionConfig]):
    """
    Generates descriptions of image chunks that contain page images with vision model.

    Descriptions are saved into chunk's metadata and can be used to build indexes to enable search by them.

    The target field will contain list of objects with the following fields:
    * `text`: generated description text
    * `kind`: the category of described item the text came from (page_summary, image_summary etc.)
    * `element_index`: for the image and table kinds, the number of that image or table inside

    The snippet bellow illustrates how generated descriptions can be referenced in indexes:

    ```json
    {
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

    @inject
    def __init__(self, config: PageDescriptionConfig, model_provider: ModelProvider = NotImplemented):
        super().__init__(config)

        self._llm = model_provider.get_llm(config.llm)

    def apply(self, chunks: AsyncIterable[AnyChunk]) -> AsyncIterable[AnyChunk]:
        return self._transform(chunks)

    async def _transform(self, chunks: AsyncIterable[AnyChunk]) -> AsyncIterable[AnyChunk]:
        semaphore = Semaphore(self.config.max_concurrency)

        async for batch in batched_async(chunks, batch_size=10):
            tasks = []
            async with TaskGroup() as task_group:
                for chunk in batch:
                    if isinstance(chunk, ImageChunk) and chunk.image_type == ImageType.page:
                        tasks.append(
                            task_group.create_task(self._create_page_description_task(semaphore, chunk))
                        )
                    else:
                        yield chunk

            for task in tasks:
                yield task.result()

    async def _create_page_description_task(self, semaphore: Semaphore, chunk: ImageChunk) -> ImageChunk:
        async with semaphore:
            description = await self._get_page_description(chunk)

        metadata = chunk.metadata.model_copy(
            update={
                self.config.field: TypeAdapter(list[DescriptionItem]).dump_python(
                    description.flatten(),
                ),
            }
        )
        return chunk.model_copy(update={"metadata": metadata})

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
