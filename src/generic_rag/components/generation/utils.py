import logging
import re

from aidial_sdk.chat_completion import Choice
from langchain_core.output_parsers import BaseTransformOutputParser, StrOutputParser
from pydantic import BaseModel

from generic_rag.utils.profile import timed_stage

REF_PATTERN = re.compile(r"<\[(\d+)\]>")

logger = logging.getLogger(__name__)


class RawLlmOutputReporter(StrOutputParser):
    def __init__(self, choice: Choice, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._choice = choice

    async def atransform(self, *args, **kwargs):
        with timed_stage(self._choice, "[DEBUG] raw llm output") as stage:
            stage.append_content("```text\n")
            async for chunk in super().atransform(*args, **kwargs):
                stage.append_content(chunk)
                yield chunk
            stage.append_content("\n```\n")


class ReferenceParserOutput(BaseModel):
    content: str | None = None
    reference: int | None = None


class ReferencesParser(BaseTransformOutputParser[ReferenceParserOutput]):
    def parse(self, text: str):
        return text

    async def atransform(self, *args, **kwargs):
        # Variable to catch pieces of document links in different chunks, like this
        # "first chunk <["; "1]> second chunk"
        prev_piece = ""

        async for chunk in super().atransform(*args, **kwargs):
            answer_piece = prev_piece + chunk
            last_pos = 0
            for m in REF_PATTERN.finditer(answer_piece):
                chunk_id = int(m.group(1))

                # id in model response is starting from 1
                chunk_index = chunk_id - 1

                yield ReferenceParserOutput(
                    content=answer_piece[last_pos : m.start()],
                    reference=chunk_index,
                )

                last_pos = m.end()

            pos = answer_piece.find("<[", last_pos)

            if pos == -1:
                pos = len(answer_piece) - 1 if answer_piece and answer_piece[-1] == "<" else len(answer_piece)

            yield ReferenceParserOutput(content=answer_piece[last_pos:pos])

            prev_piece = answer_piece[pos:]

        if prev_piece:
            yield ReferenceParserOutput(content=prev_piece)
