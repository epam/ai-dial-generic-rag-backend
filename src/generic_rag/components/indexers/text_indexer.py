import logging
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable

import jsonpath_ng
from pydantic import BaseModel, Field

from generic_rag.types import (
    AnyChunk,
    ChunkIndexer,
    IndexRecord,
    IndexRecordMeta,
    StringIndexer,
    TextChunk,
    TextType,
    VectorType,
)
from generic_rag.utils.profile import log_execution_time

logger = logging.getLogger(__name__)


class TextIndexerConfig(BaseModel):
    target: str | None = Field(
        default=None,
        description=(
            "JSON-path expression within chunk's metadata.\n\n"
            "If set, will index text elements extracted with provided expression, "
            "otherwise will index the content of text chunks."
        ),
        pattern=r"^\$(?:\.[a-zA-Z_][a-zA-Z0-9_*]*|\?|\[(?:[0-9*]+|'[^']+'|\"[^\"]+\")\])*$",
        examples=["$.description[*].text"],
    )
    concatenate_fields: bool = Field(
        default=False,
        description=(
            "Controls if values extracted with JSON-path expression (set by `target`) "
            "should be concatenated in single text element before indexing (if `true` - "
            "will emit single index record per chunk, otherwise all extracted values will "
            "be emitted as separate index records referring the same chunk)."
        ),
    )


class TextIndexer[IndexT: TextType | VectorType, ConfigT: TextIndexerConfig = TextIndexerConfig](
    ChunkIndexer[IndexT, ConfigT],
    StringIndexer[IndexT, ConfigT],
    ABC,
):
    """Indexer with common logic of text indexing."""

    def __init__(self, config: ConfigT):
        super().__init__(config)

        self._target_path: jsonpath_ng.JSONPath | None = (
            jsonpath_ng.parse(self.config.target) if self.config.target else None
        )

    @log_execution_time(logger)
    async def index_chunks(
        self, data: Iterable[tuple[AnyChunk, IndexRecordMeta]]
    ) -> Collection[IndexRecord[IndexT]]:
        texts = []
        record_metas = []

        for chunk, meta in data:
            if self._target_path:
                matched_values = [
                    match.value
                    for match in self._target_path.find(chunk.metadata.model_dump())
                    if isinstance(match.value, str) and len(match.value)
                ]
                if matched_values and self.config.concatenate_fields:
                    matched_values = ["\n".join(matched_values)]
                for text in matched_values:
                    texts.append(text)
                    record_metas.append(
                        meta.model_copy(
                            update={"text": text},
                        )
                    )

            elif isinstance(chunk, TextChunk) and chunk.text:
                texts.append(chunk.text)
                record_metas.append(meta)

        return await self._index_texts(texts, record_metas)

    @log_execution_time(logger)
    async def index_strings(
        self, data: Iterable[tuple[str, IndexRecordMeta]]
    ) -> Collection[IndexRecord[IndexT]]:
        texts = []
        record_metas = []

        for text, meta in data:
            texts.append(text)
            record_metas.append(meta)

        return await self._index_texts(texts, record_metas)

    @abstractmethod
    async def _index_texts(
        self, texts: list[str], record_metas: list[IndexRecordMeta]
    ) -> Collection[IndexRecord[IndexT]]:
        """Index given texts for further storage."""
