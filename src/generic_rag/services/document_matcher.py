import datetime
import enum
import logging
from abc import ABC
from collections.abc import Sequence
from enum import StrEnum
from operator import itemgetter
from typing import Annotated, Any, Literal

from injection import inject
from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo
from sqlalchemy import DATE, ColumnElement, Select, and_, bindparam, column, func, or_, select, true

from generic_rag.db.entities import DocumentEntity
from generic_rag.db.session import get_current_session
from generic_rag.services.metadata_service import (
    MetadataService,
    is_date_field,
    is_string_array_field,
    is_string_field,
)

logger = logging.getLogger(__name__)


@enum.unique
class SortOrder(StrEnum):
    asc = "asc"
    desc = "desc"


class TopNDocumentsModel[FieldNameT: str](BaseModel, ABC):
    """Sort documents by given fields and pick top N of them."""

    sort_by: list[FieldNameT] = Field(
        default_factory=list,
        min_length=1,
        description="List of metadata fields to sort documents by.",
    )
    order: SortOrder = Field(
        default=SortOrder.desc,
        description=f"Sorting order (`{SortOrder.asc.name}`: from low to high, `{SortOrder.desc.name}`: from high to low)",
    )
    limit: int = Field(
        default=5,
        ge=1,
        description="Maximum number of documents to select after sorting.",
    )

    @classmethod
    @inject
    async def get_dynamic_model(
        cls, metadata_service: MetadataService = NotImplemented
    ) -> type["TopNDocumentsModel"] | None:
        """Create dynamic model for TopN documents."""
        if sortable_fields := metadata_service.get_sortable_fields():
            # noinspection PyTypeHints
            return create_model(
                cls.__name__,
                __base__=cls[Literal[tuple(map(itemgetter(0), sortable_fields))]],
                __doc__=cls.__doc__,
            )
        return None


class DateInterval(BaseModel):
    """Date interval definition."""

    start: datetime.date | Literal["latest"] = None
    end: datetime.date | Literal["latest"] = None

    @property
    def latest(self) -> bool:
        return self.start == "latest" or self.end == "latest"


class EnableFilterMarker:
    """Marker class for fields that should be enabled for filtering."""


class StringValueFilterMarker(EnableFilterMarker):
    """Marker class for fields that should be filtered as strings."""


class StringArrayValueFilterMarker(EnableFilterMarker):
    """Marker class for fields that should be filtered as string arrays."""


class SingleFilterModel(BaseModel, ABC):
    """
    Configuration of a single metadata filter.
    To be matching, the document should satisfy **ALL** criteria specified here.
    """

    model_config = ConfigDict(extra="forbid")

    @classmethod
    @inject
    async def get_dynamic_model(
        cls, metadata_service: MetadataService = NotImplemented
    ) -> type["SingleFilterModel"]:
        """Create dynamic model fields available for filtering."""
        dimensions = await metadata_service.get_filtering_dimensions()
        model_keys = {}

        for key, field_info in metadata_service.get_filterable_fields():
            description = f"Match documents whose `{key}` value satisfies the criteria."

            if is_date_field(field_info):
                model_keys[key] = (
                    Annotated[DateInterval, EnableFilterMarker],
                    Field(default=None, description=description),
                )

            elif is_string_field(field_info):
                # noinspection type-hints,bad-argument-type
                value_type = Literal[tuple(dimensions.get(key))] if dimensions.get(key) else str
                model_keys[key] = (
                    Annotated[
                        value_type,
                        StringValueFilterMarker,
                    ],
                    Field(default=None, description=description),
                )

            elif is_string_array_field(field_info):
                # noinspection PyTypeHints,bad-argument-type
                value_type = Literal[tuple(dimensions.get(key))] if dimensions.get(key) else str
                model_keys[key] = (
                    Annotated[
                        value_type,
                        StringArrayValueFilterMarker,
                    ],
                    Field(default=None, description=description),
                )

        return create_model(cls.__name__, **model_keys, __base__=cls, __doc__=cls.__doc__)


class DocumentMatcherConfig[FilterT: SingleFilterModel, TopNDocumentsT](BaseModel, ABC):
    """Configuration for document matcher."""

    filters: list[FilterT] = Field(
        default_factory=list,
        description=(
            "List of metadata filters. When defined, search will be performed within documents "
            "that match to **ANY** of given filters. If empty - will search within **ALL** documents."
        ),
    )
    top_n: TopNDocumentsT | None = Field(
        default=None,
        description="Search only within top N documents sorted by given fields after applying all filters.",
    )

    @classmethod
    async def get_dynamic_model(cls) -> type["DocumentMatcherConfig"]:
        """Create dynamic model."""
        single_filter_model = await SingleFilterModel.get_dynamic_model()
        top_n_model = await TopNDocumentsModel.get_dynamic_model()

        # noinspection bad-return,bad-index
        return create_model(
            cls.__name__,
            __base__=cls[single_filter_model, top_n_model],
            __doc__=cls.__doc__,
        )

    @classmethod
    async def get_default_value(cls) -> "DocumentMatcherConfig":
        model = await cls.get_dynamic_model()
        return model.model_validate({})


class DocumentMatcher:
    """Component for getting subset of documents matching to different criteria."""

    def __init__(self, channel_key: str, config: DocumentMatcherConfig):
        self._channel_key = channel_key
        self._config = config

    async def get_documents_subset(self) -> Sequence[int] | None:
        """Return IDs of matching documents, or `None` which indicates that all documents should be used."""
        if (query := self.get_query()) is not None:
            return (await get_current_session().scalars(query)).all()
        return None

    def get_query(self) -> Select[tuple[int]] | None:
        """Get SQL query returning IDs of matching documents."""
        filtering_clauses = [
            clause
            for entry in self._config.filters
            if (clause := self._get_filter_entry_clause(entry)) is not true()
        ]
        if len(filtering_clauses) > 1:
            return self._apply_top_n_documents(or_(*filtering_clauses))
        if len(filtering_clauses) == 1:
            return self._apply_top_n_documents(filtering_clauses[0])
        return self._apply_top_n_documents(true())

    def _get_filter_entry_clause(self, filter_entry: SingleFilterModel) -> ColumnElement[bool]:
        result_clauses = []
        latest_fields = []

        for field_name, field_type in filter_entry.model_fields.items():
            if not any(issubclass(cls, EnableFilterMarker) for cls in field_type.metadata):
                continue

            value = getattr(filter_entry, field_name)

            if isinstance(value, DateInterval) and value.latest:
                latest_fields.append(field_name)

            elif (clause := self._get_field_filtering_clause(field_name, value, field_type)) is not None:
                result_clauses.append(clause)

        result_clauses.extend(
            self._get_field_max_value_clause(field_name, result_clauses) for field_name in latest_fields
        )

        if len(result_clauses) == 1:
            return result_clauses[0]
        if len(result_clauses) > 1:
            return and_(*result_clauses)
        return true()

    def _get_field_filtering_clause(
        self, name: str, value: Any, field_info: FieldInfo
    ) -> ColumnElement[bool] | None:
        key = bindparam(name, name)

        if isinstance(value, DateInterval):
            if isinstance(value.start, datetime.date) and isinstance(value.end, datetime.date):
                return func.cast(
                    DocumentEntity.metadata_[key].astext,
                    DATE,
                ).between(
                    bindparam(name, value.start, unique=True),
                    bindparam(name, value.end, unique=True),
                )

            if isinstance(value.start, datetime.date):
                return func.cast(
                    DocumentEntity.metadata_[key].astext,
                    DATE,
                ) >= bindparam(name, value.start, unique=True)

            if isinstance(value.end, datetime.date):
                return func.cast(
                    DocumentEntity.metadata_[key].astext,
                    DATE,
                ) <= bindparam(name, value.end, unique=True)

        elif isinstance(value, str):
            if any(issubclass(cls, StringValueFilterMarker) for cls in field_info.metadata):
                return DocumentEntity.metadata_[key].astext == bindparam(name, value, unique=True)

            if any(issubclass(cls, StringArrayValueFilterMarker) for cls in field_info.metadata):
                return DocumentEntity.document_id.in_(
                    select(DocumentEntity.document_id)
                    .select_from(
                        func.jsonb_array_elements_text(DocumentEntity.metadata_[key]).alias(f"{name}_element")
                    )
                    .where(
                        DocumentEntity.channel_key == self._channel_key,
                        column(f"{name}_element", is_literal=True) == bindparam(name, value, unique=True),
                    )
                )

        return None

    def _get_field_max_value_clause(
        self, name: str, filter_clauses: list[ColumnElement[bool]]
    ) -> ColumnElement[bool]:
        """
        Create a new clause for a field that will match
        its maximum value selected by applying given filter clauses.
        """
        key = bindparam(name, name)
        return DocumentEntity.metadata_[key].astext.in_(
            select(func.max(DocumentEntity.metadata_[key].astext))
            .where(DocumentEntity.channel_key == self._channel_key, *filter_clauses)
            .having(func.max(DocumentEntity.metadata_[key].astext).is_not(None))
        )

    def _apply_top_n_documents(self, clause: ColumnElement[bool]) -> Select[tuple[int]] | None:
        if not self._config.top_n and clause is true():
            return None

        if not self._config.top_n:
            return select(DocumentEntity.document_id).where(
                DocumentEntity.channel_key == self._channel_key, clause
            )

        keys = [bindparam(field, field) for field in self._config.top_n.sort_by]
        order_by = [DocumentEntity.metadata_[key] for key in keys]

        match self._config.top_n.order:
            case SortOrder.asc:
                order_by = [curr.asc() for curr in order_by]
            case SortOrder.desc:
                order_by = [curr.desc() for curr in order_by]

        return (
            select(DocumentEntity.document_id)
            .where(
                DocumentEntity.channel_key == self._channel_key,
                clause,
                *(DocumentEntity.metadata_[key].is_not(None) for key in keys),
            )
            .order_by(*order_by)
            .limit(bindparam("top_n", self._config.top_n.limit))
        )
