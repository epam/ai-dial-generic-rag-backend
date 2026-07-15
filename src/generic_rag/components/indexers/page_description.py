import asyncio
import io
import logging
import time
from collections.abc import Collection, Iterable
from functools import cached_property
from typing import Any, Literal, Self

import json_repair
from datauri import DataURI
from injection import inject
from langchain_community.callbacks import OpenAICallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from generic_rag.types import (
    AnyChunk,
    ImageChunk,
    ImageType,
    IndexedEntityMeta,
    Indexer,
    IndexRecord,
    LlmConfig,
    ModelProvider,
    VectorType,
)
from generic_rag.utils.profile import log_execution_time

logger = logging.getLogger(__name__)

PAGE_DESCRIPTION_PROMPT_TEMPLATE = """
Please create detailed description of provided image.
Ignore page header, footer, basic logo and background.
Describe all images (illustration), tables.
Text with bullet points is NOT a table or image.

Use only provided information.
DO NOT make up answer.

Provide answer in JSON format with fields:
{{
    "page_summary": "page summary here",
    "key_fact"     : "the most important fact from the image",
    "images":[
        {{
            "description": "image description",
            "type"       : "image type (photo, illustration, diagram, etc.)",
            "key_fact"    : "the most important fact from the image"
        }}
    ],
    "tables":[
        {{
            "description": "table description",
            "key_fact"    : "the most important fact from the table"
        }}
    ]
}}
"""

PAGE_DESCRIPTION_DEFAULT_LLM_DEPLOYMENT = "gpt-4.1-mini-2025-04-14"
PAGE_DESCRIPTION_MAX_IMAGE_SIZE = 800


class ImageDescription(BaseModel):
    description: str
    key_fact: str


class TableDescription(BaseModel):
    description: str
    key_fact: str


class PageDescription(BaseModel):
    page_summary: str
    key_fact: str
    images: list[ImageDescription]
    tables: list[TableDescription]

    model_config = ConfigDict(
        hide_input_in_errors=True,
    )

    @classmethod
    def create(cls, json_str: str) -> Self:
        json_page = json_repair.loads(json_str)
        assert isinstance(json_page, dict)

        images: list[dict[str, Any]] = []
        tables: list[dict[str, Any]] = []

        for image_json in json_page["images"]:
            if "image" in image_json:
                image_description = image_json["image"]["description"]
                image_key_fact = image_json["image"]["key_fact"]
            else:
                image_description = image_json["description"]
                image_key_fact = image_json["key_fact"]

            if "no images are present" in image_description.lower():
                continue

            images.append({
                "description": image_description,
                "key_fact": image_key_fact,
            })

        for table_json in json_page["tables"]:
            if "table" in table_json:
                table_description = table_json["table"]["description"]
                table_key_fact = table_json["table"]["key_fact"]
            else:
                table_description = table_json["description"]
                table_key_fact = table_json["keyfact"]

            if "no tables are present" in table_description.lower():
                continue

            tables.append({
                "description": table_description,
                "key_fact": table_key_fact,
            })

        return cls.model_validate({
            "page_summary": json_page.get("page_summary"),
            "key_fact": json_page.get("key_fact"),
            "images": images,
            "tables": tables,
        })

    @cached_property
    def texts(self) -> list[str]:
        result: list[str] = []

        def _add_to_result(chunk: str):
            result.append(chunk.replace("\n", " ").replace("\r", " "))

        _add_to_result(self.page_summary)
        _add_to_result(self.key_fact)

        for image in self.images:
            _add_to_result(image.description)
            _add_to_result(image.key_fact)

        for table in self.tables:
            _add_to_result(table.description)
            _add_to_result(table.key_fact)

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
        self, data: Iterable[tuple[AnyChunk | str, IndexedEntityMeta]]
    ) -> Collection[IndexRecord[VectorType]]:
        texts: list[str] = []
        record_metas: list[IndexedEntityMeta] = []

        for chunk, meta in data:
            if isinstance(chunk, ImageChunk) and chunk.image_type == ImageType.page:
                page_description = await self._get_page_description(chunk)
                texts.extend(page_description.texts)
                record_metas.extend([meta] * len(page_description.texts))

        embeddings = await self._embeddings.aembed_documents(texts)
        return [
            IndexRecord(index=index, metadata=meta)
            for index, meta in zip(embeddings, record_metas, strict=True)
        ]

    async def _get_page_description(self, chunk: ImageChunk) -> PageDescription:
        assert chunk.image_type == ImageType.page

        prompt = await asyncio.to_thread(self._build_prompt, chunk)

        cb = OpenAICallbackHandler()
        start_time = time.perf_counter()

        response = await self._llm.ainvoke(
            input=[HumanMessage(prompt)],
            config=RunnableConfig(callbacks=[cb]),
        )

        end_time = time.perf_counter()

        logger.debug(f"LLM Time ({self._llm}): {end_time - start_time:.2f}s")
        logger.debug(f"LLM Response: {response.content}")
        logger.debug(f"{cb.total_tokens=} ({cb.prompt_tokens=}, {cb.completion_tokens=})")

        return PageDescription.create(self._get_fixed_json(response.content))

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

            logger.info(f"{target_width=}, {target_height=}")
            image.resize(size=(target_width, target_height))

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

    @staticmethod
    def _get_fixed_json(text: str) -> str:
        text = text.replace(", ]", "]").replace(",]", "]").replace(",\n]", "]")

        # check if JSON is in code block
        if "```json" in text:
            open_bracket = text.find("```json")
            close_bracket = text.rfind("```")
            if open_bracket != -1 and close_bracket != -1:
                return text[open_bracket + 7 : close_bracket].strip()

        # check if JSON is in brackets
        tmp_text = text.replace("{", "[").replace("}", "]")
        open_bracket = tmp_text.find("[")
        if open_bracket == -1:
            return text

        close_bracket = tmp_text.rfind("]")
        if close_bracket == -1:
            return text

        return text[open_bracket : close_bracket + 1]
