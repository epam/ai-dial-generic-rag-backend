import enum
from dataclasses import dataclass
from enum import StrEnum

from injection import MappedScope
from pydantic import SecretStr

type RequestApiKey = SecretStr
type DialApplicationId = str


@enum.unique
class ScopeName(StrEnum):
    channel = enum.auto()


@dataclass
class ChannelBindings:
    """Channel scope binding."""

    request_api_key: RequestApiKey
    application_id: DialApplicationId | None

    scope = MappedScope(ScopeName.channel)
