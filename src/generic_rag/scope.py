import enum
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated

from aidial_sdk import HTTPException
from aidial_sdk.deployment.from_request_mixin import FromRequestDeploymentMixin
from fastapi import Depends, Header
from fastapi.security import APIKeyHeader
from injection import MappedScope
from pydantic import SecretStr
from starlette.status import HTTP_403_FORBIDDEN

type RequestApiKey = SecretStr
type DialApplicationId = str


@enum.unique
class ScopeName(StrEnum):
    channel = enum.auto()


@dataclass
class ChannelBindings:
    """Channel scope bindings."""

    request_api_key: RequestApiKey
    application_id: DialApplicationId

    scope = MappedScope(ScopeName.channel)

    @classmethod
    async def fastapi_auth_dep(
        cls,
        api_key: Annotated[
            str,
            Depends(
                APIKeyHeader(
                    name="Api-Key",
                    scheme_name="Api-Key",
                    description="Authorization with DIAL api key",
                )
            ),
        ],
        dial_application_id: Annotated[str, Header(alias="x-dial-application-id", include_in_schema=False)],
    ):
        """FastApi dependency to use in Application Route handlers."""
        async with cls(SecretStr(api_key), dial_application_id).scope.adefine():
            yield

    @classmethod
    @asynccontextmanager
    async def from_dial_deployment_request(cls, request: FromRequestDeploymentMixin):
        """Context manager which initializes scope for given DIAL deployment request."""
        if not request.dial_application_id:
            raise HTTPException(
                message="Application ID is not set",
                status_code=HTTP_403_FORBIDDEN,
            )
        async with cls(SecretStr(request.api_key), request.dial_application_id).scope.adefine():
            yield
