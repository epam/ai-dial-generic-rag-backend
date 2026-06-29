from collections.abc import AsyncIterator

import pytest

from generic_rag.components.generation.utils import ReferenceParserOutput, ReferencesParser


async def _achunks(chunks: list[str]) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk


async def _collect(chunks: list[str]) -> list[tuple[str | None, int | None]]:
    parser = ReferencesParser()
    return [
        (out.content, out.reference)
        async for out in parser.atransform(_achunks(chunks))
    ]


@pytest.mark.parametrize("text", ["", "anything", "Hello <[1]> world"])
def test_parse_returns_text_unchanged(text):
    assert ReferencesParser().parse(text) == text


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        pytest.param([], [], id="empty-input"),
        pytest.param([""], [("", None)], id="single-empty-chunk"),
        pytest.param(
            ["Plain text only"],
            [("Plain text only", None)],
            id="plain-text",
        ),
        # id=1 maps to index 0; a trailing empty content chunk follows the match
        pytest.param(["<[1]>"], [("", 0), ("", None)], id="just-reference"),
        pytest.param(
            ["Hello <[1]> world"],
            [("Hello ", 0), (" world", None)],
            id="one-ref-single-chunk",
        ),
        pytest.param(
            ["Hello <[1]> and <[2]> end"],
            [("Hello ", 0), (" and ", 1), (" end", None)],
            id="two-refs-single-chunk",
        ),
        pytest.param(
            ["<[12]> rest"],
            [("", 11), (" rest", None)],
            id="multi-digit-index-offset",
        ),
        pytest.param(
            ["<[1]><[2]>"],
            [("", 0), ("", 1), ("", None)],
            id="consecutive-refs",
        ),
        pytest.param(
            ["Hello <[", "1]> world"],
            [("Hello ", None), ("", 0), (" world", None)],
            id="split-at-brackets",
        ),
        pytest.param(
            ["Hello <[1", "]> world"],
            [("Hello ", None), ("", 0), (" world", None)],
            id="split-mid-digits",
        ),
        pytest.param(
            ["Hello <", "[1]> world"],
            [("Hello ", None), ("", 0), (" world", None)],
            id="split-at-less-than",
        ),
        pytest.param(
            ["Hello <", "[1", "]> world"],
            [("Hello ", None), ("", None), ("", 0), (" world", None)],
            id="split-across-three-chunks",
        ),
        # "<" not followed by "[" is treated as ordinary content
        pytest.param(
            ["abc < def"],
            [("abc < def", None)],
            id="stray-less-than-not-bracket",
        ),
        # trailing "<" is held back as a potential reference start
        pytest.param(
            ["abc <"],
            [("abc ", None), ("<", None)],
            id="chunk-ends-with-less-than",
        ),
        # "<[" with no closing "]>" gets buffered and finally emitted as content
        pytest.param(
            ["abc <[partial"],
            [("abc ", None), ("<[partial", None)],
            id="unterminated-bracket-flushed-at-end",
        ),
    ],
)
async def test_atransform_yields_expected_outputs(chunks, expected):
    assert await _collect(chunks) == expected


async def test_atransform_yields_reference_parser_output_instances():
    parser = ReferencesParser()
    outputs = [out async for out in parser.atransform(_achunks(["Hi <[1]>!"]))]
    assert all(isinstance(o, ReferenceParserOutput) for o in outputs)
