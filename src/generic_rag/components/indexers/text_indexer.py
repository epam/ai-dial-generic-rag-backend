import logging
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable

from pydantic import BaseModel

from generic_rag.types import (
    AnyChunk,
    Indexer,
    IndexRecord,
    IndexRecordMeta,
    TextChunk,
    TextType,
    VectorType,
)
from generic_rag.utils.profile import log_execution_time

logger = logging.getLogger(__name__)


class TextIndexer[IndexT: TextType | VectorType, ConfigT: BaseModel = BaseModel](
    Indexer[IndexT, ConfigT], ABC
):
    """Indexer with common logic of text data indexing."""

    @log_execution_time(logger)
    async def index_data(
        self, data: Iterable[tuple[AnyChunk | str, IndexRecordMeta]]
    ) -> Collection[IndexRecord[IndexT]]:
        texts = []
        record_metas = []

        for item, meta in data:
            if (text := self._get_text(item)) is not None:
                texts.append(text)
                record_metas.append(meta)

        return await self._index_texts(texts, record_metas)

    @staticmethod
    def _get_text(item: str | AnyChunk) -> str | None:
        match item:
            case str():
                return item
            case TextChunk():
                return item.text
        return None

    @abstractmethod
    async def _index_texts(
        self, texts: list[str], record_metas: list[IndexRecordMeta]
    ) -> Collection[IndexRecord[IndexT]]:
        """Index given texts for further storage."""
