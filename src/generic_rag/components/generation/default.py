import logging
from typing import Any

from injection import inject
from langchain_core.documents import Document as LangchainDocument
from langchain_core.messages import BaseMessage, HumanMessage, merge_content
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from generic_rag.channel import Channel
from generic_rag.components.generation.prompts import DefaultGenerationPrompt
from generic_rag.components.generation.utils import ReferencesParser
from generic_rag.services.chunk_sources_manager import ChunkSource
from generic_rag.types import (
    AnswerCallback,
    AnswerGenerator,
    AnyChunk,
    ImageChunk,
    LlmConfig,
    ModelProvider,
    Retriever,
    TextChunk,
)

logger = logging.getLogger(__name__)


def _text_element(text: str) -> dict:
    return {"type": "text", "text": text}


def _image_element(image_url: str) -> dict:
    return {
        "type": "image_url",
        "image_url": {
            "url": image_url
        }
    }


def _format_attributes(i: int, chunk: AnyChunk, source_attributes: list[str]) -> str:
    chunk_source = ChunkSource.from_chunk(chunk)
    attributes = [
        ("id", i),
        ("source", chunk_source.source_url),
        ("page_number", chunk.page_number),
        ("title", chunk_source.source_display_name),
    ]
    if source_attributes:
        attributes += [(name, str(chunk_source.source_metadata.get(name, ""))) for name in source_attributes]
    return " ".join(
        [f"{key}='{value}'" for key, value in attributes if value is not None]
    )


class DefaultChatPromptChainInputSchema(BaseModel):
    query: str
    found_items: list[LangchainDocument]


class DefaultAnswerGeneratorConfig(BaseModel):
    """ Configuration for the chat chain which generates the answer for the user question. """
    llm: LlmConfig = Field(
        default=LlmConfig(),
        description="Configuration for the LLM used in the query chain.",
    )
    system_prompt_template_override: str | SkipJsonSchema[None] = Field(
        default=None,
        description="Allow to override the system prompt template.",
    )
    metadata_instructions: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Additional instructions for LLM about document's metadata fields. When specified, these "
            "instructions and corresponding metadata properties will be added to default system prompt."
        )
    )


class DefaultChatPromptChain(Runnable[DefaultChatPromptChainInputSchema, list[BaseMessage]]):
    """ A chain that creates messages to be sent in LLM. """

    def __init__(self, generation_config: DefaultAnswerGeneratorConfig, metadata_schema: dict[str, Any]):
        self._generation_config = generation_config
        self._source_attributes = []

        document_metadata_schema = metadata_schema

        if self._generation_config.system_prompt_template_override:
            self._system_prompt_template = self._generation_config.system_prompt_template_override
        elif self._generation_config.metadata_instructions and document_metadata_schema:
            extra_llm_notes = []

            for key, value in self._generation_config.metadata_instructions.items():
                if key in document_metadata_schema.get("properties", {}):
                    self._source_attributes.append(key)
                    extra_llm_notes.append(value)

            self._system_prompt_template = DefaultGenerationPrompt.get_prompt(
                extra_llm_notes
            )
        else:
            self._system_prompt_template = DefaultGenerationPrompt.get_prompt()

    def invoke(self, *args, **kwargs) -> list[BaseMessage]:
        raise NotImplementedError()

    async def ainvoke(self, chain_input: DefaultChatPromptChainInputSchema, *args, **kwargs) -> list[BaseMessage]:
        docs_message = await self._create_docs_message(chain_input.found_items)

        template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self._system_prompt_template),
                HumanMessagePromptTemplate.from_template("{query}"),
            ]
        )

        prompt_messages = template.invoke(chain_input).to_messages()
        assert len(prompt_messages) > 1

        last_message = prompt_messages[-1]
        assert isinstance(last_message, HumanMessage)
        assert isinstance(last_message.content, str)

        merged_content = merge_content([_text_element(last_message.content)], docs_message)

        prompt_messages[-1] = HumanMessage(content=merged_content)
        return prompt_messages

    async def _create_docs_message(self, found_items: list[LangchainDocument]) -> list[dict[str, Any]]:
        result = [_text_element("<context>")]

        for i, document in enumerate(found_items, start=1):
            chunks: list[AnyChunk] = document.metadata.get("chunks", [])
            assert len(chunks) > 0

            attributes = _format_attributes(i, chunks[0], self._source_attributes)
            images = []
            content = ""

            for chunk in chunks:
                if isinstance(chunk, TextChunk):
                    content += f"\n{chunk.text}"

                elif isinstance(chunk, ImageChunk):
                    images.append(chunk.get_data_uri())

            result.append(_text_element(f"<doc {attributes}>{content}\n"))
            result.extend(_image_element(image) for image in images)
            result.append(_text_element("</doc>\n"))

        result.append(_text_element("</context>"))

        return result


class DefaultAnswerGenerator(AnswerGenerator[DefaultAnswerGeneratorConfig]):
    """ Generates answer with LLM based on chunks returned by configured retriever. """

    @inject
    def __init__(self, config: DefaultAnswerGeneratorConfig, channel: Channel, model_provider: ModelProvider):
        super().__init__(config)
        self._channel = channel
        self._model_provider = model_provider

    async def invoke(self, query: str, retriever: Retriever, callback: AnswerCallback):
        """
        Generate answer to given user's query.

        :param query: the user query to answer
        :param retriever: the :class:`Retriever` used to find relevant chunk information
        :param callback: a callback to catch answer as it is generated
        """
        llm = self._model_provider.get_llm(
            self.config.llm
        )
        found_items: list[LangchainDocument] = await retriever.invoke(query)

        generation_chain = (
            DefaultChatPromptChain(self.config, self._channel.metadata_schema)
            | llm
            | StrOutputParser()
            | ReferencesParser()
        )

        chain_input = DefaultChatPromptChainInputSchema(
            query=query,
            found_items=found_items,
        )

        async for item in generation_chain.astream(chain_input):
            if item.content:
                callback.append_content(item.content)
            if item.reference is not None:
                reference_index = item.reference
                if not (0 <= reference_index < len(found_items)):
                    logger.warning(
                        f"Reference idx in model response is out of bounds: {reference_index} / {len(found_items)}"
                    )
                    continue
                callback.append_reference(reference_index, found_items[reference_index])
