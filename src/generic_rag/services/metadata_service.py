import datetime
import logging
import typing
from collections.abc import Generator
from types import GenericAlias

import json_schema_to_pydantic
from injection import scoped
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from sqlalchemy import column, func, select

from generic_rag.channel import Channel
from generic_rag.db.entities import DocumentEntity
from generic_rag.db.session import get_current_session, transaction
from generic_rag.scope import ScopeName

ENABLE_FILTERING_MARKER = "enable_filtering"
ENABLE_IN_MCP_RETRIEVE_CHUNKS_MARKER = "enable_in_mcp_retrieve_chunks"

logger = logging.getLogger(__name__)


def is_string_field(field_info: FieldInfo) -> bool:
    return type(field_info.annotation) is type and issubclass(field_info.annotation, str)


def is_date_field(field_info: FieldInfo) -> bool:
    return type(field_info.annotation) is type and issubclass(field_info.annotation, datetime.date)


def is_string_array_field(field_info: FieldInfo) -> bool:
    if hasattr(field_info.annotation, "__origin__") and hasattr(field_info.annotation, "__args__"):
        # noinspection PyInvalidCast
        generic = typing.cast(GenericAlias, field_info.annotation)
        if (
            issubclass(generic.__origin__, list)
            and len(generic.__args__) == 1
            and issubclass(generic.__args__[0], str)
        ):
            return True
    return False


@scoped(ScopeName.channel)
class MetadataService:
    """Service to work with document metadata schema."""

    def __init__(self, channel: Channel):
        self._channel_key = channel.channel_key
        self._model: type[BaseModel] = json_schema_to_pydantic.create_model(channel.metadata_schema)

    def _iter_properties(self, marker: str) -> Generator[tuple[str, FieldInfo]]:
        """Iterate over properties that have given marker set."""
        for key, field_info in self._model.model_fields.items():
            field_info = typing.cast(FieldInfo, field_info)
            json_schema_extra = field_info.json_schema_extra
            if json_schema_extra and json_schema_extra.get(marker):
                yield key, field_info

    def get_filterable_fields(self) -> list[tuple[str, FieldInfo]]:
        """Return list of fields that can be used to filter documents."""
        return list(self._iter_properties(ENABLE_FILTERING_MARKER))

    def get_mcp_retrieve_chunks_field_names(self) -> set[str] | None:
        """
        Return names of metadata fields to include in the ``retrieve_text_chunks``
        MCP tool response. Returns ``None`` when no field carries the marker —
        callers should treat that as "include all metadata" for back-compat.
        """
        names = {name for name, _ in self._iter_properties(ENABLE_IN_MCP_RETRIEVE_CHUNKS_MARKER)}
        return names or None

    def get_sortable_fields(self) -> list[tuple[str, FieldInfo]]:
        """Return list of fields that can be used to sort documents."""
        return [
            (key, field_info)
            for key, field_info in self.get_filterable_fields()
            if (is_string_field(field_info) or is_date_field(field_info))
        ]

    @transaction
    async def get_filtering_dimensions(self) -> dict[str, list[str]]:
        """Get "dimensions" (lists of possible values) for metadata fields"""
        result: dict[str, list[str]] = {}
        string_dimensions = []

        for key, field_info in self.get_filterable_fields():
            if is_string_field(field_info):
                string_dimensions.append(key)
            elif is_string_array_field(field_info) and (
                array_values := await self._get_array_field_dimensions(key)
            ):
                result[key] = array_values

        if string_dimensions:
            result.update(await self._get_string_fields_dimensions(string_dimensions))

        return result

    async def _get_string_fields_dimensions(self, names: list[str]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}

        cursor = await get_current_session().execute(
            select(
                column("js.key", is_literal=True).label("key"),
                column("js.value", is_literal=True).label("value"),
            )
            .distinct()
            .select_from(
                DocumentEntity,
                func.jsonb_each(DocumentEntity.metadata_).alias(
                    "js",
                ),
            )
            .where(
                DocumentEntity.channel_key == self._channel_key, column("js.key", is_literal=True).in_(names)
            )
            .order_by("key", "value")
        )

        for name, value in cursor.all():
            result.setdefault(name, []).append(value)

        return result

    async def _get_array_field_dimensions(self, field_name: str) -> list[str]:
        cursor = await get_current_session().scalars(
            select(
                column("value", is_literal=True),
            )
            .distinct()
            .select_from(
                DocumentEntity,
                func.jsonb_array_elements_text(DocumentEntity.metadata_[field_name]).alias("value"),
            )
            .where(
                DocumentEntity.channel_key == self._channel_key,
            )
        )

        return list(cursor.all())
