from collections.abc import Collection

from generic_rag.components.indexers.text_indexer import TextIndexer
from generic_rag.types import IndexRecord, IndexRecordMeta, TextType


class TextNoopIndexer(TextIndexer[TextType]):
    """Pass source text to the index storage without any modifications."""

    async def index_query(self, query: str) -> TextType:
        return query

    async def _index_texts(
        self, texts: list[str], record_metas: list[IndexRecordMeta]
    ) -> Collection[IndexRecord[TextType]]:
        return [
            IndexRecord(index=text, metadata=meta) for text, meta in zip(texts, record_metas, strict=True)
        ]
