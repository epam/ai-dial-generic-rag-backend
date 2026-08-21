import asyncio
import copy
import json
import logging
from abc import ABC
from functools import cached_property
from typing import Annotated, Any, Self

import jsonschema.validators
from annotated_types import MinLen
from fastapi.encoders import jsonable_encoder
from injection import afind_instance
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    conlist,
    create_model,
    field_validator,
)
from pydantic.config import JsonDict
from pydantic.fields import FieldInfo
from pydantic.v1.utils import deep_update
from pydantic_core import InitErrorDetails, PydanticCustomError

from generic_rag.components.search_index import ChunkIndex, Index, IndexConfig
from generic_rag.types import (
    AnswerGenerator,
    DocumentParser,
    Retriever,
)

logger = logging.getLogger(__name__)

METADATA_SCHEMA_EXAMPLE = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "DocumentMetadataSchemaExample",
    "type": "object",
    "properties": {
        "publication_type": {"type": "string", "enable_filtering": True},
        "publication_date": {"type": "string", "format": "date", "enable_filtering": True},
        "publication_topics": {"type": "array", "items": {"type": "string"}, "enable_filtering": True},
    },
    "additionalProperties": True,
}


class ProcessingConfig(BaseModel, ABC):
    """Options related to processing pipeline of data stored in the channel."""

    metadata_schema: dict[str, Any] = Field(
        create_model("DefaultMetadataSchema", __config__=ConfigDict(extra="allow")).model_json_schema(),
        description="JSON schema of metadata that can be associated with documents of this channel",
        examples=[METADATA_SCHEMA_EXAMPLE],
    )
    parsers: list[BaseModel]
    indexes: dict[str, IndexConfig]

    @field_validator("metadata_schema", mode="after")
    @classmethod
    def validate_metadata_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        object_validator_cls = jsonschema.validators.validator_for(value)
        schema_validator_cls = jsonschema.validators.validator_for(
            object_validator_cls.META_SCHEMA,
            default=object_validator_cls,
        )
        schema_validator = schema_validator_cls(
            schema=object_validator_cls.META_SCHEMA,
            format_checker=schema_validator_cls.FORMAT_CHECKER,
        )

        if line_errors := [
            InitErrorDetails(
                type=PydanticCustomError(
                    "json_schema_error",
                    "JSON Schema: {message} {schema}",
                    {
                        "message": error.message,
                        "schema": json.dumps(error.schema) if isinstance(error.schema, dict) else "",
                    },
                ),
                loc=tuple(error.absolute_path),
                input=error.instance,
            )
            for error in schema_validator.iter_errors(value)
        ]:
            raise ValidationError.from_exception_data(title="ValidationError", line_errors=line_errors)

        return value

    @classmethod
    async def get_dynamic_model[T: ProcessingConfig](cls: type[T]) -> type[T]:
        document_parser_config_model = await DocumentParser.get_aggregated_config_model()
        index_config_model = await Index.get_aggregated_config_model(default_impl=ChunkIndex)

        # noinspection PyTypeHints,PyTypeChecker
        return create_model(
            cls.__name__,
            __base__=cls,
            __doc__=cls.__doc__,
            parsers=Annotated[
                conlist(document_parser_config_model, min_length=1),  # type: ignore
                Field(
                    ...,
                    description=(
                        "List of parsers to use for extracting chunks from documents uploaded to the channel."
                    ),
                ),
            ],
            indexes=Annotated[
                dict[
                    Annotated[
                        str,
                        Field(
                            description="Internal name of the index.",
                            examples=["keyword", "semantic-index"],
                        ),
                    ],
                    index_config_model,
                ],
                MinLen(1),
                Field(description="Configuration of indexes for relevance search."),
            ],
        )


class RequestConfig(BaseModel, ABC):
    """Options related to a request to a channel (like, data retrieval or answering)."""

    retriever: BaseModel
    generation: BaseModel

    @classmethod
    async def get_dynamic_model[T: RequestConfig](cls: type[T]) -> type[T]:
        retriever_config_model = await Retriever.get_aggregated_config_model()

        generation_config_model = await AnswerGenerator.get_aggregated_config_model()
        generation_config_model_default = TypeAdapter(generation_config_model).validate_python({
            "type": "default"
        })

        # noinspection PyTypeChecker
        return create_model(
            cls.__name__,
            __base__=cls,
            __doc__=cls.__doc__,
            retriever=retriever_config_model,
            generation=Annotated[
                generation_config_model,
                Field(
                    default=generation_config_model_default,
                    description="Configuration for the chat chain which generates the answer for the user question.",
                ),
            ],
        )

    @classmethod
    def create(cls, *, defaults: Self, overrides: dict[str, Any] | None) -> Self:
        # we need to perform model_dump -> model_validate in all cases even if `overrides` is empty
        # because target (`cls`) model may be dynamic and may contain extra fields which are missing
        # in `defaults`, but we still want the result to contain default values for these fields
        return cls.model_validate(
            deep_update(
                defaults.model_dump(exclude_none=True),
                overrides or {},
            )
        )


class ChannelConfig(RequestConfig, ProcessingConfig, ABC):
    """Channel configuration schema."""

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        for i, field_info in enumerate(cls.model_fields.values()):
            assert isinstance(field_info, FieldInfo)
            property_meta: JsonDict = {
                "dial:meta": {
                    "dial:propertyKind": "server",
                    "dial:propertyOrder": i,
                }
            }
            if isinstance(field_info.json_schema_extra, dict):
                field_info.json_schema_extra.update(**property_meta)
            else:
                field_info.json_schema_extra = property_meta
        cls.model_rebuild(force=True)

    @classmethod
    async def get_dynamic_model[T: ChannelConfig](cls: type[T]) -> type[T]:
        # noinspection PyTypeChecker
        return create_model(
            ChannelConfig.__name__,
            __doc__=ChannelConfig.__doc__,
            __base__=(
                await RequestConfig.get_dynamic_model(),
                await ProcessingConfig.get_dynamic_model(),
                cls,
            ),
        )  # type: ignore


class Channel:
    """Logical concept that represents instance of DIAL application."""

    @classmethod
    async def get_current_channel(cls) -> Self:
        return await afind_instance(cls)

    def __init__(self, channel_key: str, channel_config: ChannelConfig):
        self._channel_key = channel_key
        self._config = channel_config
        self._indexes: list[ChunkIndex] | None = None
        self._lock = asyncio.Lock()

    @property
    def channel_key(self):
        """Unique key of the channel."""
        return self._channel_key

    @property
    def metadata_schema(self):
        """JSON schema of metadata that can be associated with documents of this channel."""
        return copy.deepcopy(self._config.metadata_schema)

    @cached_property
    def document_parsers(self) -> list[DocumentParser]:
        """Document parsers configured for this channel."""
        result = []
        for config in self._config.parsers:
            if parser := DocumentParser.create(config):
                result.append(parser)
            else:
                logger.warning(
                    "unable to find document parser for configuration: " + config.model_dump_json(indent=2)
                )
        return result

    async def get_indexes(self) -> list[ChunkIndex]:
        """Get indexes configured for this channel."""
        async with self._lock:
            if self._indexes is None:
                tasks = [
                    Index.create_async(config, channel_key=self._channel_key, index_name=name)
                    for name, config in self._config.indexes.items()
                ]
                self._indexes = list(await asyncio.gather(*tasks))

        assert self._indexes is not None
        return self._indexes

    @cached_property
    def request_config(self) -> RequestConfig:
        """Default request configuration for this channel."""
        # looking for dynamic RequestConfig model that was used to validate the config
        request_config_model = next(
            candidate
            for candidate in type(self._config).mro()
            if (issubclass(candidate, RequestConfig) and not issubclass(candidate, ChannelConfig))
        )
        return request_config_model.model_validate(self._config.model_dump(exclude_none=True))

    def dump_config(self) -> dict[str, Any]:
        """Create a dictionary representation of this channel's configuration."""
        return jsonable_encoder(
            dict(
                channel_key=self._channel_key,
                **self._config.model_dump(
                    exclude_none=True,
                ),
            )
        )
