import logging
from collections.abc import AsyncIterable, Collection, Iterable

from elasticsearch import AsyncElasticsearch, helpers
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from generic_rag.types import (
    IndexedEntityMeta,
    Indexer,
    IndexerCompatibilityError,
    IndexRecord,
    IndexStorage,
    IndexStorageBackend,
    TextType,
    VectorType,
)
from generic_rag.utils.profile import log_execution_time

logger = logging.getLogger(__name__)


class SearchHitTotal(BaseModel):
    """Metadata about the number of matching documents."""

    value: int = Field(description="Total number of matching documents.")
    relation: str = Field(
        description="Indicates whether the number of matching documents in the value parameter is accurate or a lower bound."
    )

    model_config = ConfigDict(populate_by_name=True)


class SearchHit(BaseModel):
    index: str = Field(alias="_index", description="Name of the index containing the returned document.")
    id: str = Field(
        alias="_id",
        description="Unique identifier for the returned document. This ID is only unique within the returned index.",
    )
    score: float | None = Field(
        alias="_score",
        description="Positive 32-bit floating point number used to determine the relevance of the returned document.",
    )
    source: dict = Field(
        alias="_source",
        description="Original JSON body passed for the document at index time.",
    )
    highlight: dict | None = Field(
        default=None,
        description=(
            "Contains highlighted snippets from the returned document."
            " This field is only present if highlighting is requested."
        ),
    )

    model_config = ConfigDict(populate_by_name=True)


class SearchHits(BaseModel):
    total: SearchHitTotal
    max_score: float | None = Field(
        description="Highest returned document _score. This value is `None` for requests that do not sort by _score."
    )
    hits: list[SearchHit]

    model_config = ConfigDict(populate_by_name=True)


class SearchResult(BaseModel):
    took: int = Field(description="Milliseconds it took Elasticsearch to execute the request.")
    timed_out: bool = Field(
        description="If true, the request timed out before completion; returned results may be partial or empty."
    )
    hits: SearchHits = Field(description="Contains returned documents and metadata.")
    aggregations: dict | None = Field(
        default=None, description="Contains aggregation results if aggregations were requested."
    )
    scroll_id: str | None = Field(alias="_scroll_id", default=None, description="Scroll identifier.")

    model_config = ConfigDict(populate_by_name=True)


class ElasticsearchIndexStorageOptions(BaseModel):
    """Storage configuration for Elasticsearch indexes."""


class ElasticsearchIndexStorage[IndexT: TextType](IndexStorage[IndexT]):
    """Storage implementation for text indexes stored in ElasticSearch."""

    def __init__(self, index: str, client: AsyncElasticsearch):
        """
        :param index: name of corresponding index in Elasticsearch
        :param client: instance of ElasticSearch client to use
        """
        self._index = index
        self._client = client

    @log_execution_time(logger)
    async def relevance_search(
        self,
        query: IndexT,
        limit: int,
        documents: Collection[int] | None = None,
    ) -> Collection[IndexedEntityMeta]:
        """
        Search for entities that are relevant to the given query.

        :param query: the indexed query used for search
        :param limit: maximum number of results to return
        :param documents: if set, the scope of search will be limited only to given documents
        :return: collection of the most relevant entities metadata (sorted by relevancy)
        """
        if documents is not None and len(documents) < 1:
            return []

        if not await self._client.indices.exists(index=self._index):
            return []

        search_query = {
            "match": {
                "content": query,
            }
        }

        if documents:
            search_query = {
                "bool": {
                    "filter": {"terms": {"metadata.document_id": list(documents)}},
                    "should": search_query,
                }
            }

        raw_result = await self._client.search(index=self._index, query=search_query, size=limit)
        result = SearchResult.model_validate(raw_result.body)

        return [IndexedEntityMeta.model_validate(hit.source["metadata"]) for hit in result.hits.hits]

    async def add(self, records: Iterable[IndexRecord[IndexT]]):
        """
        Add given index records to the storage.

        :param records: record to update index with
        """
        if not await self._client.indices.exists(index=self._index):
            await self._client.indices.create(index=self._index)

        actions = (
            {
                "_index": self._index,
                "content": record.index,
                "metadata": record.metadata.model_dump(mode="json"),
            }
            for record in records
        )
        await helpers.async_bulk(self._client, actions)

    async def remove(self, *documents: int):
        """Remove index records for documents with given IDs."""
        if not await self._client.indices.exists(index=self._index):
            return

        await self._client.delete_by_query(
            index=self._index,
            query={
                "terms": {
                    "metadata.document_id": list(documents),
                }
            },
        )

    def export(self, *documents: int) -> AsyncIterable[IndexRecord[IndexT]]:
        """Export index records for documents with given IDs."""
        return self._get_all_records(*documents)

    @log_execution_time(logger)
    async def _get_all_records(self, *documents: int):
        if not await self._client.indices.exists(index=self._index):
            return

        async for raw_result in helpers.async_scan(
            client=self._client,
            index=self._index,
            query={
                "query": {
                    "terms": {"metadata.document_id": list(documents)},
                }
            },
        ):
            hit = SearchHit.model_validate(raw_result)
            yield IndexRecord.model_validate({
                "index": hit.source["content"],
                "metadata": hit.source["metadata"],
            })


class ElasticsearchIndexStorageBackend[StorageOptionsT: ElasticsearchIndexStorageOptions](
    IndexStorageBackend[StorageOptionsT]
):
    """Storage backend that stores indexes in Elasticsearch."""

    def __init__(self, client: AsyncElasticsearch, index_prefix: str):
        self._client = client
        self._index_prefix = (index_prefix or "").rstrip("-")

    @classmethod
    def get_qualifier(cls) -> str:
        return "elastic"

    async def get_storage[IndexT: TextType | VectorType](
        self,
        channel_key: str,
        index_name: str,
        indexer: Indexer[IndexT, BaseModel],
        options: StorageOptionsT | None = None,
    ) -> IndexStorage[IndexT]:
        """
        Return index storage for given channel, index name and indexer.

        :param channel_key: the key of the channel
        :param index_name: name of the index (should be unique within the channel)
        :param indexer: the indexer object
        :param options: additional options for the storage (depend on implementation)
        """
        index = f"{channel_key}-{index_name}"
        if self._index_prefix:
            index = f"{self._index_prefix}-{index}"

        sample = await indexer.index_query("Lorem ipsum dolor sit amet.")

        if TypeAdapter(TextType).validator.isinstance_python(sample):
            return ElasticsearchIndexStorage(index, self._client)

        raise IndexerCompatibilityError(self, indexer)

    @property
    def storage_options_model(self) -> type[StorageOptionsT]:
        return ElasticsearchIndexStorageOptions  # type: ignore
