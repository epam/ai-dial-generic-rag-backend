import datetime
import re

import pytest

from generic_rag.components.generation.default import (
    DefaultAnswerGeneratorConfig,
    DefaultChatPromptChain,
    DefaultChatPromptChainInputSchema,
)
from generic_rag.types import AnyChunk, ImageChunk, ImageType, RetrievedDocument, TextChunk

SOURCE_URL = "files/BUCKET-ID/appdata/generic-rag-deployment/reports/Some%20Report%202024.pdf"
DISPLAY_NAME = "Some Report 2024.pdf"

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def _text_chunk(text: str, *, chunk_id: int = 1, page_number: int = 1) -> TextChunk:
    return TextChunk(
        document_id=1,
        chunk_id=chunk_id,
        page_number=page_number,
        text=text,
    )


def _image_chunk(chunk_id: int = 2, page_number: int = 1) -> ImageChunk:
    return ImageChunk(
        document_id=1,
        chunk_id=chunk_id,
        page_number=page_number,
        image_type=ImageType.page,
        mime_type="image/png",
        content=PNG,
    )


def _document(*chunks: AnyChunk, **source) -> RetrievedDocument:
    return RetrievedDocument(
        chunks=list(chunks),
        source_id=chunks[0].document_id,
        source_page_number=chunks[0].page_number,
        source_url=source.get("source_url", SOURCE_URL),
        source_display_name=source.get("source_display_name", DISPLAY_NAME),
        source_metadata=source.get("source_metadata", {}),
    )


async def _build(
    query: str,
    documents: list[RetrievedDocument],
    *,
    metadata_schema: dict | None = None,
    **config,
):
    chain = DefaultChatPromptChain(DefaultAnswerGeneratorConfig(**config), metadata_schema or {})
    return await chain.ainvoke(DefaultChatPromptChainInputSchema(query=query, found_items=documents))


def _human_text(messages) -> str:
    """Concatenate the text elements of the final human message."""
    return "".join(el["text"] for el in messages[-1].content if el["type"] == "text")


def _query_element(messages) -> dict:
    """The element carrying the query - the one right after the current date."""
    return messages[-1].content[1]


async def test_query_appears_exactly_once_and_on_its_own():
    query = "What was the deficit in 2024?"
    messages = await _build(query, [_document(_text_chunk("CHUNK_BODY"))])

    assert _query_element(messages) == {"type": "text", "text": f"<query>{query}</query>"}
    assert _human_text(messages).count(query) == 1


async def test_current_date_precedes_the_query():
    messages = await _build("q", [_document(_text_chunk("CHUNK_BODY"))])
    prompt = _human_text(messages)

    today = datetime.datetime.now(datetime.UTC).date().isoformat()
    assert prompt.startswith(f"<current_date>{today}</current_date><query>")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", today)


async def test_query_precedes_the_context_block():
    messages = await _build("q", [_document(_text_chunk("CHUNK_BODY"))])
    prompt = _human_text(messages)

    assert "<query>q</query><context>" in prompt


async def test_no_input_schema_repr_leaks_into_the_prompt():
    """Regression test for the model being stringified into the '{query}' slot."""
    messages = await _build("q", [_document(_text_chunk("CHUNK_BODY"))])
    prompt = _human_text(messages)

    assert "found_items=" not in prompt
    assert "Document(" not in prompt
    assert "metadata=" not in prompt


async def test_each_chunk_body_appears_exactly_once():
    documents = [
        _document(_text_chunk("FIRST_BODY", chunk_id=1)),
        _document(_text_chunk("SECOND_BODY", chunk_id=2)),
    ]
    prompt = _human_text(await _build("q", documents))

    assert prompt.count("FIRST_BODY") == 1
    assert prompt.count("SECOND_BODY") == 1


async def test_storage_path_is_not_exposed_to_the_model():
    messages = await _build("q", [_document(_text_chunk("CHUNK_BODY"))])
    prompt = _human_text(messages)

    assert SOURCE_URL not in prompt
    assert "files/" not in prompt
    assert "BUCKET-ID" not in prompt
    assert "source=" not in prompt


async def test_doc_attributes_expose_only_the_display_name():
    prompt = _human_text(await _build("q", [_document(_text_chunk("CHUNK_BODY"))]))

    assert f"<doc id='1' document='{DISPLAY_NAME}' page_number='1'>" in prompt
    assert "document_id" not in prompt
    assert "chunk_id" not in prompt


async def test_metadata_instructions_add_configured_source_attributes():
    chunk = _text_chunk("CHUNK_BODY")
    prompt = _human_text(
        await _build(
            "q",
            [_document(chunk, source_metadata={"year": "2024"})],
            metadata_instructions={"year": "The reporting year."},
            metadata_schema={"properties": {"year": {"type": "string"}}},
        )
    )

    assert "year='2024'" in prompt


async def test_image_chunks_reach_the_payload_in_openai_block_shape():
    messages = await _build("q", [_document(_text_chunk("CHUNK_BODY"), _image_chunk())])

    images = [el for el in messages[-1].content if el["type"] == "image_url"]
    assert len(images) == 1
    assert set(images[0]) == {"type", "image_url"}
    assert set(images[0]["image_url"]) == {"url"}
    assert images[0]["image_url"]["url"].startswith("data:image/png;base64,")


async def test_doc_ids_are_one_based_positions_in_found_items():
    """The citation contract: '<[N]>' resolves to found_items[N-1]."""
    documents = [_document(_text_chunk(f"BODY_{i}", chunk_id=i)) for i in range(1, 4)]
    prompt = _human_text(await _build("q", documents))

    for position, document in enumerate(documents, start=1):
        body = document.chunks[0].text
        assert f"<doc id='{position}'" in prompt
        assert prompt.index(f"<doc id='{position}'") < prompt.index(body)


async def test_system_prompt_is_first_and_query_is_not_repeated_there():
    query = "UNIQUE_QUERY_TOKEN"
    messages = await _build(query, [_document(_text_chunk("CHUNK_BODY"))])

    system_message, _human_message = messages
    assert system_message.type == "system"
    assert isinstance(system_message.content, str)
    assert query not in system_message.content


@pytest.mark.parametrize(
    "override",
    [
        "A fixed system prompt.",
        'Reply as JSON: {"answer": "..."} and cite sources.',
        "Braces {like this} and {{these}} are literal.",
        "Placeholders such as {query} are not substituted.",
    ],
    ids=["plain", "json-example", "braces", "query-placeholder"],
)
async def test_system_prompt_override_is_used_verbatim(override):
    """The override is prose, not a template - nothing is substituted into it."""
    messages = await _build(
        "UNIQUE_QUERY_TOKEN", [_document(_text_chunk("CHUNK_BODY"))], system_prompt_template_override=override
    )

    assert messages[0].content == override


@pytest.mark.parametrize(
    "query",
    ["", "braces {in} the query", "{query}", "multi\nline", "quote ' and \" chars"],
)
async def test_query_is_never_reinterpreted_as_a_template(query):
    messages = await _build(query, [_document(_text_chunk("CHUNK_BODY"))])

    assert _query_element(messages) == {"type": "text", "text": f"<query>{query}</query>"}
