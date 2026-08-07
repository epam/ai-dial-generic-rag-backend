import pytest
from langchain_core.language_models.chat_models import _format_for_tracing
from langchain_core.messages import HumanMessage, SystemMessage

from generic_rag.utils.llm import LCMessageLogger, _redact_content_block

PAYLOAD = "iVBORw0KGgo" * 40
DATA_URI = f"data:image/png;base64,{PAYLOAD}"


def _image_url_block(**image_url) -> dict:
    return {"type": "image_url", "image_url": {"url": DATA_URI, **image_url}}


def test_image_payload_is_replaced_by_its_size():
    redacted = _redact_content_block(_image_url_block())

    assert redacted["image_url"]["url"] == f"data:image/png;base64,<{len(PAYLOAD)} chars redacted>"
    assert PAYLOAD not in str(redacted)


def test_reported_size_is_the_payload_not_the_whole_uri():
    """The 'data:...;base64,' prefix must not be counted."""
    redacted = _redact_content_block(_image_url_block())

    assert f"<{len(PAYLOAD)} chars redacted>" in redacted["image_url"]["url"]
    assert f"<{len(DATA_URI)} chars redacted>" not in redacted["image_url"]["url"]


def test_sibling_keys_of_the_image_block_are_preserved():
    """'detail' is set by the page-description indexer; the log must keep showing it."""
    redacted = _redact_content_block(_image_url_block(detail="high"))

    assert redacted["image_url"]["detail"] == "high"
    assert redacted["type"] == "image_url"


def test_non_image_blocks_pass_through_untouched():
    block = {"type": "text", "text": "some prompt text"}

    assert _redact_content_block(block) == block
    assert _redact_content_block("a bare string") == "a bare string"


def test_unknown_url_shape_does_not_raise():
    redacted = _redact_content_block({"type": "image_url", "image_url": {}})

    assert redacted["image_url"]["url"] == "data:?;base64,<0 chars redacted>"


def test_v1_image_block_is_redacted_if_it_ever_reaches_the_logger():
    block = {"type": "image", "base64": PAYLOAD, "mime_type": "image/png"}
    redacted = _redact_content_block(block)

    assert redacted["base64"] == f"<{len(PAYLOAD)} chars redacted>"
    assert redacted["mime_type"] == "image/png"


def test_v1_image_blocks_are_normalised_before_the_callback_sees_them():
    """Guards the assumption behind the 'image' branch being a safeguard only."""
    block = {"type": "image", "base64": PAYLOAD, "mime_type": "image/png"}
    traced = _format_for_tracing([HumanMessage(content=[block])])[0].content[0]

    assert traced["type"] == "image_url"
    assert _redact_content_block(traced)["image_url"]["url"].endswith("chars redacted>")


def test_image_url_blocks_survive_format_for_tracing_unchanged():
    """'detail' really does reach the logger, so preserving it is not hypothetical."""
    block = _image_url_block(detail="high")

    assert _format_for_tracing([HumanMessage(content=[block])])[0].content[0] == block


@pytest.mark.parametrize(
    "content",
    [
        f"inline {DATA_URI} then more text",
        f"at the very end: {DATA_URI}",
        f"{DATA_URI}\nfollowed by a newline",
        f"quoted '{DATA_URI}'",
    ],
    ids=["followed-by-text", "end-of-string", "before-newline", "quoted"],
)
def test_data_uri_in_string_content_is_redacted_regardless_of_what_follows(content):
    result = LCMessageLogger.langchain_msg_2_role_content(SystemMessage(content=content))

    assert PAYLOAD not in result["content"]
    assert "data:image/png;base64,<base64_image>" in result["content"]


def test_list_content_is_redacted_block_by_block():
    message = HumanMessage(content=[{"type": "text", "text": "q"}, _image_url_block()])
    result = LCMessageLogger.langchain_msg_2_role_content(message)

    assert result["role"] == "human"
    assert result["content"][0] == {"type": "text", "text": "q"}
    assert PAYLOAD not in str(result)


def test_redaction_does_not_mutate_the_message_sent_to_the_model():
    block = _image_url_block(detail="high")
    message = HumanMessage(content=[block])

    LCMessageLogger.langchain_msg_2_role_content(message)

    assert message.content[0]["image_url"]["url"] == DATA_URI
