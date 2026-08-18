import logging
import os
import re
import sys
from typing import Self

from pydantic import (
    BaseModel,
    ByteSize,
    Field,
    HttpUrl,
    SecretStr,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)


class DatabaseConfig(BaseModel):
    """Configuration for database connection."""

    host: str = Field(description="Postgresql database host")
    port: int = Field(default=5432, description="Postgresql database port")
    dbname: str = Field(description="Postgresql database name")
    username: str = Field(description="Postgresql database username")
    password: SecretStr | None = Field(
        None, description="Database password, if you plan to use password authentication"
    )
    msi_enabled: bool = Field(
        default=False,
        description="Use MSI authentication for database access",
    )

    @model_validator(mode="after")
    def validate_auth(self) -> Self:
        if not (self.password or self.msi_enabled):
            raise ValueError("either 'password' or 'msi_enabled' should be set")
        return self


class ElasticsearchSettings(BaseModel):
    """Configuration for Elasticsearch connection."""

    url: HttpUrl
    username: str
    password: SecretStr
    index_prefix: str | None = None

    @field_validator("index_prefix", mode="after")
    @classmethod
    def validate_index_prefix(cls, value: str | None):
        if value is None:
            return None

        errors: list[str] = []

        # just a magic number that should be sufficient: index name cannot be longer than 255 bytes,
        # and we need to reserve some space for actual index name (prefixed by a value)
        max_length = 64

        # validate value using some rules described here:
        # https://www.elastic.co/guide/en/elasticsearch/reference/8.19/indices-create-index.html
        if value.lower() != value:
            errors.append("can be lowercase only")
        if re.search(r"[\\/*?\"<>|\s,#:]", value):
            errors.append('cannot include any of `\\ / * ? " < > | , # :` or space character')
        if value.startswith(("-", "_", "+")):
            errors.append("cannot start with `-`, `_` or `+`")
        if value in {".", ".."}:
            errors.append("cannot be `.` or `..`")
        if len(value.encode("utf-8")) > max_length:
            errors.append(f"cannot be longer than {max_length} bytes")

        if errors:
            raise ValueError(", ".join(errors))
        return value


class InMemoryCacheSettings(BaseModel):
    """Configuration for in-memory files caching."""

    enabled: bool = True
    capacity: ByteSize = Field(
        default=TypeAdapter(ByteSize).validate_python("128MiB"),
        description=(
            "Used to cache the file contents and avoid requesting Dial Core File API every time, "
            "if user makes several requests for the same document. Could be increased to reduce load "
            "on the Dial Core File API if we have a lot of concurrent users "
            "(requires corresponding increase of the pod memory). "
            "Could be integer for bytes, or a pydantic.ByteSize compatible string (e.g. 128MiB, 1GiB, 2.5GiB)."
        ),
    )


class ApplicationSettings(BaseModel):
    """Main application settings class."""

    dial_url: HttpUrl = Field(description="URL to the DIAL core.")
    dial_api_key: SecretStr | None = Field(description="optional api-key for background jobs execution")
    dial_public_url: HttpUrl | None = Field(
        None,
        description="URL where DIAL core is publicly accessible (used to generate interactive documentation).",
    )
    in_memory_cache: InMemoryCacheSettings = InMemoryCacheSettings()
    database: DatabaseConfig = Field(..., description="Configuration for postgres/vector database connection")
    elasticsearch: ElasticsearchSettings | None = Field(None, description="Elasticsearch settings")
    enable_debug_stages: bool = Field(False, description="enable debug stages reporting")


def get_app_settings() -> ApplicationSettings:
    raw_config = {
        "dial_url": os.environ.get("DIAL_URL"),
        "dial_api_key": os.environ.get("DIAL_API_KEY"),
        "dial_public_url": os.environ.get("DIAL_PUBLIC_URL"),
        "in_memory_cache": {
            "enabled": os.environ.get("IN_MEMORY_CACHE_ENABLED", "yes"),
            "capacity": os.environ.get("IN_MEMORY_CACHE_CAPACITY", "128MiB"),
        },
        "database": {
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT", "5432"),
            "dbname": os.getenv("DB_NAME"),
            "username": os.getenv("DB_USERNAME"),
            "password": os.getenv("DB_PASSWORD"),
            "msi_enabled": os.getenv("DB_MSI_ENABLED", "no"),
        },
        "elasticsearch": {
            "url": os.getenv("ELASTICSEARCH_URL"),
            "username": os.getenv("ELASTICSEARCH_USERNAME"),
            "password": os.getenv("ELASTICSEARCH_PASSWORD"),
            "index_prefix": os.getenv("ELASTICSEARCH_INDEX_PREFIX"),
        }
        if os.getenv("ELASTICSEARCH_URL")
        else None,
        "enable_debug_stages": os.getenv("ENABLE_DEBUG_STAGES", "no"),
    }
    try:
        return ApplicationSettings.model_validate(raw_config)
    except ValidationError as e:
        logger.error(str(e))
        sys.exit(-1)
