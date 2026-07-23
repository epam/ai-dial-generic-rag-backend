import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Callable, Collection, Iterable
from contextlib import AbstractAsyncContextManager
from typing import Annotated, Any, Literal, Self

from async_lru import alru_cache
from deepdiff import DeepDiff
from pgvector.sqlalchemy import VECTOR
from pydantic import BaseModel, Field, TypeAdapter, create_model
from sqlalchemy import (
    INTEGER,
    Column,
    ColumnElement,
    Index,
    PrimaryKeyConstraint,
    String,
    Table,
    Text,
    bindparam,
    delete,
    func,
    insert,
    select,
    text,
    true,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncScalarResult, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql.ddl import DDL, CreateIndex, CreateTable
from sqlalchemy.sql.schema import SchemaItem
from sqlalchemy.util import OrderedSet
from tenacity import retry, retry_if_exception_type, stop_after_attempt

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

_schema_lock = asyncio.Lock()

DbSessionFactory = Callable[..., AbstractAsyncContextManager[AsyncSession]]

logger = logging.getLogger(__name__)


async def _is_table_exist(session: AsyncSession, table_name: str) -> bool:
    return await session.scalar(
        text("select exists(select * from information_schema.tables where table_name=:table_name)"),
        params={"table_name": table_name},
    )


class _EntityBase(DeclarativeBase): ...


class IndexEntity(_EntityBase):
    """Information about indexes that exist in the system."""

    __tablename__ = "_indexes"

    channel_key: Mapped[str] = mapped_column(String, nullable=False)
    index_name: Mapped[str] = mapped_column(String, nullable=False)
    table_name: Mapped[str] = mapped_column(String, nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default={})

    __table_args__ = (PrimaryKeyConstraint("channel_key", "index_name"),)

    @classmethod
    @alru_cache()
    async def ensure_table_exists(cls, session_factory: DbSessionFactory):
        async with session_factory() as session, _schema_lock:
            if await _is_table_exist(session, cls.__tablename__):
                return
            logger.info(f"creating table '{cls.__tablename__}' in database")
            assert isinstance(cls.__table__, Table)
            await session.execute(
                CreateTable(cls.__table__, if_not_exists=True),
            )


class TableIndexStorage[IndexT: TextType | VectorType](IndexStorage[IndexT], ABC):
    """Base class for :class:`IndexStorage` implementations that stores index data as postgresql table."""

    def __init__(self, session_factory: DbSessionFactory, channel_key: str, index_name: str):
        """
        :param session_factory: callable that returns database session
        :param channel_key: the key of the channel
        :param index_name: name of the index (should be unique within the channel)
        """
        self._session_factory: DbSessionFactory = session_factory
        self._channel_key = channel_key
        self._index_name = index_name

    @log_execution_time(logger)
    async def add(self, records: Iterable[IndexRecord[IndexT]]):
        """
        Add given index records to the storage.

        :param records: record to update index with
        """
        table = await self._get_table()
        rows = [record.model_dump(mode="json") for record in records]

        if not rows:
            return

        async with self._session_factory() as session:
            await session.execute(insert(table), rows)

    async def remove(self, *documents: int):
        """Remove index records for documents with given IDs."""
        table = await self._get_table()

        async with self._session_factory() as session:
            await session.execute(
                delete(table).where(self._get_documents_filtering_clause(table, *documents))
            )

    def export(self, *documents: int) -> AsyncIterable[IndexRecord[IndexT]]:
        """Export index records for documents with given IDs."""
        return self._get_all_records(*documents)

    @staticmethod
    def _get_documents_filtering_clause(table: Table, *document_ids: int) -> ColumnElement[bool]:
        """Return expression to use in `where` clause to apply filtering of given documents."""
        return func.cast(
            table.c.metadata["document_id"].astext,
            INTEGER,
        ).in_(document_ids)

    async def _get_all_records(self, *documents: int) -> AsyncIterable[IndexRecord[IndexT]]:
        table = await self._get_table()

        async with self._session_factory() as session:
            result = await session.execute(
                select(
                    table.c.index,
                    table.c.metadata,
                ).where(self._get_documents_filtering_clause(table, *documents))
            )
            for index, metadata in result:
                yield IndexRecord.model_validate({
                    "index": index,
                    "metadata": metadata,
                })

    async def _get_table(self) -> Table:
        """Returns a table with this index's data."""
        table_name = await self._get_table_name()

        async with _schema_lock:
            if table_name in _EntityBase.metadata.tables:
                return _EntityBase.metadata.tables[table_name]

            table = Table(
                table_name,
                _EntityBase.metadata,
                Column("metadata", JSONB, nullable=False),
                Index(f"idx__{table_name}__documents", text("((metadata->>'document_id')::int)")),
                *self._get_index_column_schema(table_name),
            )

            async with self._session_factory() as session:
                if not await _is_table_exist(session, table_name):
                    logger.info(f"creating table '{table_name}' in database")
                    await session.execute(
                        CreateTable(table, if_not_exists=True),
                    )
                    for idx in table.indexes:
                        await session.execute(
                            CreateIndex(idx, if_not_exists=True),
                        )

            return table

    @retry(stop=stop_after_attempt(10), retry=retry_if_exception_type(IntegrityError))
    async def _get_table_name(self) -> str:
        """Returns a name of database table with index data."""
        await IndexEntity.ensure_table_exists(self._session_factory)

        async with self._session_factory() as session, _schema_lock:
            index_entity = await session.scalar(
                select(IndexEntity).where(
                    IndexEntity.channel_key == self._channel_key,
                    IndexEntity.index_name == self._index_name,
                )
            )

            if index_entity is None:
                index_name_hash = hashlib.md5(f"{self._channel_key}/{self._index_name}".encode()).hexdigest()
                index_entity = IndexEntity(
                    channel_key=self._channel_key,
                    index_name=self._index_name,
                    table_name=f"index_{index_name_hash}",
                    metadata_=self._index_metadata,
                )
                session.add(index_entity)
                await session.flush()

            if diff := DeepDiff(index_entity.metadata_, self._index_metadata):
                raise ValueError(
                    f"metadata mismatch for table '{index_entity.table_name}': {diff.to_json(indent=2)}"
                )

            return str(index_entity.table_name)

    @property
    @abstractmethod
    def _index_metadata(self) -> dict[str, Any]:
        """Additional data that should be associated with index table."""

    @abstractmethod
    def _get_index_column_schema(self, table_name: str) -> Collection[SchemaItem]:
        """Return schema (column definition and indexes) for `index` column."""


class VectorIndexStorage(TableIndexStorage[VectorType]):
    """Storage implementation for vector indexes."""

    def __init__(
        self, session_factory: DbSessionFactory, channel_key: str, index_name: str, vector_length: int
    ):
        """
        :param session_factory: callable that returns database session
        :param channel_key: the key of the channel
        :param index_name: name of the index (should be unique within the channel)
        :param vector_length: the length of vectors stored in the table
        """
        super().__init__(session_factory, channel_key, index_name)
        self._vector_length = vector_length

    @property
    def _index_metadata(self) -> dict[str, Any]:
        return {"index_type": "vector", "vector_length": self._vector_length}

    def _get_index_column_schema(self, table_name: str) -> Collection[SchemaItem]:
        return [Column("index", VECTOR(self._vector_length), nullable=False)]

    @log_execution_time(logger)
    async def relevance_search(
        self,
        query: VectorType,
        limit: int,
        documents: Collection[int] | None = None,
    ) -> Collection[IndexedEntityMeta]:
        if documents is not None and len(documents) < 1:
            return []

        table = await self._get_table()
        result: OrderedSet[IndexedEntityMeta] = OrderedSet()
        offset = 0

        async with self._session_factory() as session:
            where_clause = (
                [self._get_documents_filtering_clause(table, *documents)] if documents is not None else []
            )
            while len(result) < limit:
                scalar_result: AsyncScalarResult = await session.scalars(
                    select(table.c.metadata)
                    .where(*where_clause)
                    .order_by(table.c.index.cosine_distance(query))
                    .offset(bindparam("offset", offset))
                    .limit(bindparam("limit", limit * 2))
                )

                if not (rows := scalar_result.all()):
                    break

                for row in rows:
                    if (meta := IndexedEntityMeta.model_validate(row)) not in result:
                        result.add(meta)
                    if len(result) >= limit:
                        break

                offset += len(rows)

        return result


class TextIndexStorageOptions(BaseModel, ABC):
    """Storage configuration for text indexes stored as postgresql tables."""

    language: str

    @classmethod
    async def get_dynamic_model(cls, session: AsyncSession) -> type["TextIndexStorageOptions"]:
        languages = tuple(await session.scalars(text("select cfgname from pg_ts_config")))
        assert "english" in languages

        # noinspection PyTypeHints
        languages_type = Literal[tuple(languages)] if languages else str

        return create_model(
            TextIndexStorageOptions.__name__,
            __base__=TextIndexStorageOptions,
            __doc__=TextIndexStorageOptions.__doc__,
            language=Annotated[
                languages_type,
                Field(
                    "english", description="Value of language to use by postgresql full-text search index."
                ),
            ],
        )


class TextIndexStorage(TableIndexStorage[TextType]):
    """Storage implementation for text indexes."""

    def __init__(
        self,
        session_factory: DbSessionFactory,
        channel_key: str,
        index_name: str,
        options: TextIndexStorageOptions,
    ):
        """
        :param session_factory: callable that returns database session
        :param channel_key: the key of the channel
        :param index_name: name of the index (should be unique within the channel)
        :param options: additional options for the storage
        """
        super().__init__(session_factory, channel_key, index_name)
        self._options = options

    @property
    def _index_metadata(self) -> dict[str, Any]:
        return {"index_type": "text", "language": self._options.language}

    def _get_index_column_schema(self, table_name: str) -> Collection[SchemaItem]:
        return [
            Column("index", Text, nullable=False),
            Index(
                f"idx__{table_name}__index",
                text(f"to_tsvector('{self._options.language}', index)"),
                postgresql_using="gin",
            ),
        ]

    @log_execution_time(logger)
    async def relevance_search(
        self,
        query: TextType,
        limit: int,
        documents: Collection[int] | None = None,
    ) -> Collection[IndexedEntityMeta]:
        if documents is not None and len(documents) < 1:
            return []

        table = await self._get_table()

        ts_vector = func.to_tsvector(self._options.language, table.c.index)
        ts_query = func.plainto_tsquery(self._options.language, query)

        async with self._session_factory() as session:
            result = await session.execute(
                select(
                    table.c.metadata,
                    func.ts_rank_cd(ts_vector, ts_query).label("rank"),
                )
                .where(
                    self._get_documents_filtering_clause(table, *documents)
                    if documents is not None
                    else true(),
                    ts_vector.bool_op("@@")(ts_query),
                )
                .order_by(text("rank desc"))
                .limit(bindparam("limit", limit))
            )
            return [IndexedEntityMeta.model_validate(raw_meta) for raw_meta, _ in result]


class PgvectorIndexStorageOptions(BaseModel):
    text: TextIndexStorageOptions

    @classmethod
    async def get_dynamic_model(cls, session: AsyncSession) -> type["PgvectorIndexStorageOptions"]:
        text_model = await TextIndexStorageOptions.get_dynamic_model(session)
        text_default = text_model.model_validate({})

        return create_model(
            cls.__name__,
            __base__=cls,
            __doc__=cls.__doc__,
            text=Annotated[
                text_model,
                Field(
                    text_default,
                    description="Options for text storage (used only if selected indexer produces text).",
                ),
            ],
        )


class PgvectorIndexStorageBackend[StorageOptionsT: PgvectorIndexStorageOptions](
    IndexStorageBackend[StorageOptionsT]
):
    """Storage backend that stores indexes as tables in postgresql/pgvector database."""

    _storage_options_model: type[PgvectorIndexStorageOptions] = PgvectorIndexStorageOptions
    _storage_options_default: PgvectorIndexStorageOptions

    @classmethod
    async def create(cls, session_factory: DbSessionFactory) -> Self:
        """Create instance of the class and perform its initialization"""
        instance = cls(session_factory)
        await instance._initialize()
        return instance

    def __init__(self, session_factory: DbSessionFactory):
        self._session_factory = session_factory

    @alru_cache()
    async def _initialize(self):
        # todo: check `vector` extension (and if it is missing - try to create it)

        async with self._session_factory() as session:
            self._storage_options_model = await PgvectorIndexStorageOptions.get_dynamic_model(session)
            self._storage_options_default = self._storage_options_model.model_validate({})

            await self._migrate_indexes(session)

    @staticmethod
    @log_execution_time(logger)
    async def _migrate_indexes(session: AsyncSession):
        if not await _is_table_exist(session, IndexEntity.__tablename__):
            return

        await _unify_index_column_names(session)
        await _ensure_metadata_column(session)

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
        sample = await indexer.index_query("Lorem ipsum dolor sit amet.")

        if TypeAdapter(TextType).validator.isinstance_python(sample):
            return TextIndexStorage(
                self._session_factory,
                channel_key,
                index_name,
                options and options.text or self._storage_options_default.text,
            )

        if TypeAdapter(VectorType).validator.isinstance_python(sample):
            return VectorIndexStorage(
                self._session_factory,
                channel_key,
                index_name,
                vector_length=len(sample),
            )

        raise IndexerCompatibilityError(self, indexer)

    @property
    def storage_options_model(self) -> type[StorageOptionsT]:
        return self._storage_options_model  # type: ignore


async def _unify_index_column_names(session: AsyncSession):
    for table_name, column_name in await session.execute(
        text(
            f"""
                select table_name, column_name
                from information_schema.columns
                where table_name in (select table_name from {IndexEntity.__tablename__})
                  and column_name in ('vector', 'text')
                """
        )
    ):
        logger.info(f"{table_name}: rename column '{column_name}'")
        await session.execute(DDL(f"alter table {table_name} rename column {column_name} to index"))


async def _ensure_metadata_column(session: AsyncSession):
    for table_name in await session.scalars(
        text(
            f"""
                select distinct table_name
                from information_schema.columns c
                where c.table_name in (select table_name from {IndexEntity.__tablename__})
                  and not exists(
                    select * from information_schema.columns sub
                    where sub.table_name = c.table_name
                      and sub.column_name = 'metadata'
                  )
                """
        )
    ):
        logger.info(f"{table_name}: migrating storage structure")

        for statement in [
            DDL(f"alter table {table_name} add column metadata jsonb"),
            DDL(
                f"update {table_name} set metadata = jsonb_build_object('document_id', document_id, 'chunk_id', chunk_id, 'chunk_type', chunk_type)"
            ),
            DDL(f"alter table {table_name} alter column metadata set not null"),
            DDL(f"drop index if exists idx__{table_name}__documents"),
            DDL(
                f"create index if not exists idx__{table_name}__documents on {table_name} (((metadata->>'document_id')::int))"
            ),
            DDL(f"alter table {table_name} drop column document_id"),
            DDL(f"alter table {table_name} drop column chunk_id"),
            DDL(f"alter table {table_name} drop column chunk_type"),
        ]:
            await session.execute(statement)
