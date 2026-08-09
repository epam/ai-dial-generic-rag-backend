import datetime
import logging
from typing import Any

from injection import inject
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from generic_rag.channel import Channel
from generic_rag.components.generation.prompts import DefaultGenerationPrompt
from generic_rag.components.generation.utils import RawLlmOutputReporter, ReferencesParser
from generic_rag.types import (
    AbstractAnswer,
    AnswerGenerator,
    ImageChunk,
    LlmConfig,
    ModelProvider,
    RetrievedDocument,
    Retriever,
    TextChunk,
)

logger = logging.getLogger(__name__)


def _text_element(text: str) -> dict:
    return {"type": "text", "text": text}


def _image_element(image_url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": image_url}}


class DefaultChatPromptChainInputSchema(BaseModel):
    query: str
    found_items: list[RetrievedDocument]


class DefaultAnswerGeneratorConfig(BaseModel):
    """Configuration for the chat chain which generates the answer for the user question."""

    llm: LlmConfig = Field(
        default=LlmConfig(),
        description="Configuration for the LLM used in the query chain.",
    )
    system_prompt_template_override: str | None = Field(
        default=None,
        description="Allow to override the system prompt template.",
    )
    metadata_instructions: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Additional instructions for LLM about document's metadata fields. When specified, these "
            "instructions and corresponding metadata properties will be added to default system prompt."
        ),
    )


class DefaultChatPromptChain[Input: DefaultChatPromptChainInputSchema, Output: list[BaseMessage]](
    Runnable[DefaultChatPromptChainInputSchema, list[BaseMessage]]
):
    """A chain that creates messages to be sent in LLM."""

    def __init__(self, generation_config: DefaultAnswerGeneratorConfig, metadata_schema: dict[str, Any]):
        self._generation_config = generation_config
        self._source_attributes = []

        document_metadata_schema = metadata_schema

        if self._generation_config.system_prompt_template_override:
            self._system_prompt = self._generation_config.system_prompt_template_override
        elif self._generation_config.metadata_instructions and document_metadata_schema:
            extra_llm_notes = []

            for key, value in self._generation_config.metadata_instructions.items():
                if key in document_metadata_schema.get("properties", {}):
                    self._source_attributes.append(key)
                    extra_llm_notes.append(value)

            self._system_prompt = DefaultGenerationPrompt.get_prompt(extra_llm_notes)
        else:
            self._system_prompt = DefaultGenerationPrompt.get_prompt()

    def invoke(self, *args, **kwargs) -> list[BaseMessage]:
        raise NotImplementedError()

    # noinspection method-overriding
    async def ainvoke(self, chain_input: Input, *args, **kwargs: Any) -> Output:
        context = await self._get_context_elements(chain_input.found_items)
        today = datetime.datetime.now(datetime.UTC).date().isoformat()

        return [
            SystemMessage(content=self._system_prompt),
            HumanMessage(
                content=[
                    _text_element(f"<current_date>{today}</current_date>"),
                    _text_element(f"<query>{chain_input.query}</query>"),
                    *context,
                ]
            ),
        ]

    async def _get_context_elements(self, found_items: list[RetrievedDocument]) -> list[dict[str, Any]]:
        result = [_text_element("<context>")]

        for i, document in enumerate(found_items, start=1):
            attributes = self._format_attributes(i, document, self._source_attributes)
            images = []
            content = ""

            for chunk in document.chunks:
                if isinstance(chunk, TextChunk):
                    content += f"\n{chunk.text}"

                elif isinstance(chunk, ImageChunk):
                    images.append(chunk.get_data_uri())

            result.append(_text_element(f"<doc {attributes}>{content}\n"))
            result.extend(_image_element(image) for image in images)
            result.append(_text_element("</doc>\n"))

        result.append(_text_element("</context>"))

        return result

    @staticmethod
    def _format_attributes(i: int, doc: RetrievedDocument, source_attributes: list[str]) -> str:
        attributes = [
            ("id", i),
            ("document", doc.source_display_name),
            ("page_number", doc.source_page_number),
        ]
        if source_attributes:
            attributes += [(name, str(doc.source_metadata.get(name, ""))) for name in source_attributes]
        return " ".join([f"{key}='{value}'" for key, value in attributes if value is not None])


class DefaultAnswerGenerator(AnswerGenerator[DefaultAnswerGeneratorConfig]):
    """Generates answer with LLM based on chunks returned by configured retriever."""

    @inject
    def __init__(self, config: DefaultAnswerGeneratorConfig, channel: Channel, model_provider: ModelProvider):
        super().__init__(config)
        self._channel = channel
        self._model_provider = model_provider

    async def invoke(self, query: str, retriever: Retriever, answer: AbstractAnswer):
        """
        Generate answer to given user's query.

        :param query: the user query to answer
        :param answer: the current answer
        :param retriever: the :class:`Retriever` used to find relevant chunk information
        """
        llm = self._model_provider.get_llm(self.config.llm)
        found_items = await retriever.invoke(query, answer)

        generation_chain = (
            DefaultChatPromptChain(self.config, self._channel.metadata_schema)
            | llm
            | StrOutputParser()
            | RawLlmOutputReporter(answer.create_stage("raw llm output", debug=True))
            | ReferencesParser()
        )

        chain_input = DefaultChatPromptChainInputSchema(
            query=query,
            found_items=found_items,
        )

        used_references: list[int] = []

        async for item in generation_chain.astream(chain_input):
            if item.content:
                answer.append_content(item.content)
            if item.reference is not None:
                if not (0 <= item.reference < len(found_items)):
                    logger.warning(
                        f"Reference idx in model response is out of bounds: {item.reference} / {len(found_items)}"
                    )
                    continue

                if item.reference not in used_references:
                    used_references.append(item.reference)
                    citation_index = len(used_references)
                    await answer.add_reference(citation_index, found_items[item.reference])
                else:
                    citation_index = used_references.index(item.reference) + 1

                await answer.add_citation(citation_index, found_items[item.reference])
