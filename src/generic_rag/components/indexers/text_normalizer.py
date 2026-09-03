import enum
import logging
from collections.abc import Collection
from enum import StrEnum
from functools import lru_cache

import spacy
from pydantic import BaseModel
from spacy.language import Language

from generic_rag.components.indexers.text_indexer import TextIndexer
from generic_rag.types import IndexRecord, IndexRecordMeta, TextType

logger = logging.getLogger(__name__)


@enum.unique
class LanguageName(StrEnum):
    english = "english"
    ukrainian = "ukrainian"


class TextNormalizerConfig(BaseModel):
    language: LanguageName = LanguageName.english


@lru_cache
def _get_pipeline(lang: LanguageName) -> Language:
    match lang:
        case lang.english:
            return spacy.load("en_core_web_sm")
        case lang.ukrainian:
            return spacy.load("uk_core_news_sm")
        case _:
            # noinspection PyUnreachableCode
            raise RuntimeError(f"Unknown language: {lang}")


class TextNormalizer(TextIndexer[TextType, TextNormalizerConfig]):
    """Perform text normalization (by removing stopwords and applying stemming)."""

    @property
    def _pipeline(self) -> Language:
        return _get_pipeline(self.config.language)

    async def index_query(self, query: str) -> TextType:
        return self._normalize_text(query)

    async def _index_texts(
        self, texts: list[str], record_metas: list[IndexRecordMeta]
    ) -> Collection[IndexRecord[TextType]]:
        return [
            IndexRecord(
                index=self._normalize_text(item),
                metadata=meta,
            )
            for item, meta in zip(texts, record_metas, strict=True)
        ]

    def _normalize_text(self, text: str):
        doc = self._pipeline(text)
        return " ".join([token.lemma_ for token in doc if not (token.is_stop or token.is_punct)])
