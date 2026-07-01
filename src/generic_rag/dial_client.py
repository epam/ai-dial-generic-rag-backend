import io
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any, BinaryIO, Self
from urllib.parse import quote, urljoin

import httpx
import pydantic
from aiohttp import ClientResponse, ClientResponseError, ClientSession, FormData
from async_lru import alru_cache
from cachetools import Cache
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import ConfigDict
from pydantic.alias_generators import to_camel
from starlette.status import HTTP_404_NOT_FOUND

from generic_rag.types import FileMetadata, FileStorage, LlmConfig, ModelProvider
from generic_rag.utils.caching import CachingFileStorage
from generic_rag.utils.llm import LCMessageLogger

FILE_CHUNK_SIZE = 512 * 1024  # 512KB
OPENAI_API_VERSION = "2023-03-15-preview"

logger = logging.getLogger(__name__)


class UserClaim(pydantic.BaseModel):
    email: str
    sub: str
    model_config = ConfigDict(frozen=True)


class UserInfo(pydantic.BaseModel):
    roles: list[str]
    project: str
    user_claims: list[UserClaim] | None = pydantic.Field(None, alias="userClaims")
    model_config = ConfigDict(frozen=True)


class DialFileMetadata(FileMetadata):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class DialFileStorage(FileStorage):
    def __init__(self, session: ClientSession, dial_url: str, api_key: str):
        self._session = session
        self._dial_url = dial_url
        self._headers = {"Api-Key": api_key}

    async def get_bucket(self) -> str:
        bucket, _ = await self._get_bucket_ids()
        return bucket

    @alru_cache()
    async def _get_bucket_ids(self) -> tuple[str, str | None]:
        """
        Return pair of (bucket, appdata).

        * the value of `bucket` is always returned and depending on the type of used api-key
          it can point to the application bucket (for per-request api-key) or a project bucket
          (for a project api-key)
        * the value of `appdata` points to a folder inside a user bucket and present only for
          requests made with per-request api-keys

        See https://docs.dialx.ai/platform/core/per-request-keys#files-sharing for more details.
        """
        request_url = urljoin(self._dial_url, "/v1/bucket")

        async with self._session.get(request_url, headers=self._headers) as response:
            try:
                response.raise_for_status()
            except ClientResponseError:
                logger.error(await response.text())
                raise

            data = await response.json()

        return data.get("bucket"), data.get("appdata")

    async def put_file(
        self, bucket: str, filepath: str, content_type: str, content: bytes | BinaryIO
    ) -> FileMetadata:
        data = FormData()
        data.add_field(
            name="attachment",
            content_type=content_type,
            value=io.BytesIO(content) if isinstance(content, bytes) else content,
            filename=filepath,
        )

        url = f"files/{bucket}/{filepath}"
        request_url = urljoin(self._dial_url, f"/v1/{url}")

        logger.debug(f"Uploading file '{url}'")

        async with self._session.put(request_url, data=data, headers=self._headers) as response:
            try:
                response.raise_for_status()
            except ClientResponseError:
                logger.error(await response.text())
                raise

            metadata = await response.json()

        return DialFileMetadata.model_validate(metadata)

    @alru_cache(ttl=5)
    async def get_file_metadata(self, url: str) -> FileMetadata | None:
        assert url.startswith("files/")

        request_url = urljoin(self._dial_url, f"/v1/metadata/{url}")

        async with self._session.get(request_url, headers=self._headers) as response:
            if response.status == HTTP_404_NOT_FOUND:
                return None

            try:
                response.raise_for_status()
            except ClientResponseError:
                logger.error(await response.text())
                raise

            return DialFileMetadata.model_validate(await response.json())

    async def download_file(self, url: str) -> AsyncIterable[bytes] | None:
        assert url.startswith("files/")

        request_url = urljoin(self._dial_url, f"/v1/metadata/{url}")

        async with self._session.get(request_url, headers=self._headers) as response:
            if response.status == HTTP_404_NOT_FOUND:
                return None

            await self._check_response(response)

        return self._get_content_stream(url, chunk_size=FILE_CHUNK_SIZE)

    async def _get_content_stream(self, url: str, *, chunk_size: int) -> AsyncGenerator[bytes]:
        request_url = urljoin(self._dial_url, f"/v1/{url}")

        async with self._session.get(request_url, headers=self._headers) as response:
            await self._check_response(response)

            logger.debug(f"Downloading file '{url}'")

            async for chunk in response.content.iter_chunked(chunk_size):
                yield chunk

    async def copy_file_to_user(self, source_url: str, destination_name: str) -> str:
        assert source_url.startswith("files/")

        bucket, appdata = await self._get_bucket_ids()

        if appdata is None:
            raise RuntimeError("appdata path is not defined")

        destination_url = f"files/{appdata}/{quote(destination_name)}"

        source_meta = await self.get_file_metadata(source_url)
        destination_meta = await self.get_file_metadata(destination_url)

        if not (source_meta and destination_meta and source_meta.etag == destination_meta.etag):
            request_url = urljoin(self._dial_url, "/v1/ops/resource/copy")

            body = {
                "sourceUrl": source_url,
                "destinationUrl": destination_url,
                "overwrite": True,
            }

            async with self._session.post(request_url, headers=self._headers, json=body) as response:
                try:
                    response.raise_for_status()
                except ClientResponseError:
                    logger.error(await response.text())
                    raise

        return destination_url

    async def delete_file(self, url: str):
        assert url.startswith("files/")

        request_url = urljoin(self._dial_url, f"/v1/{url}")

        async with self._session.delete(request_url, headers=self._headers) as response:
            if response.ok:
                logger.debug(f"Deleted file: {url}")
            elif response.status == HTTP_404_NOT_FOUND:
                logger.debug(f"File not found: {url}")
                return

            try:
                response.raise_for_status()
            except ClientResponseError:
                logger.error(await response.text())
                raise

    @staticmethod
    async def _check_response(response: ClientResponse):
        try:
            response.raise_for_status()
        except ClientResponseError:
            logger.error(await response.text())
            raise


class DialClient(ModelProvider):
    """Convenience class that encapsulates requests to DIAL."""

    def __init__(
        self, session: ClientSession, dial_url: str, api_key: str, in_memory_cache: Cache | None = None
    ):
        """
        :param session: instance of ClientSession used for making http requests
        :param dial_url: value of dial url
        :param api_key: value of dial api-key associated with this instance of the client
        :param in_memory_cache: instance of Cache used to temporary storage of file contents
        """
        self._session = session
        self._dial_url = dial_url
        self._api_key = api_key
        self._headers = {"Api-Key": api_key}
        self._in_memory_cache = in_memory_cache

    def bind(self, api_key: str) -> Self:
        """Return new instance of the client bound to given API key."""
        return DialClient(
            self._session,
            self._dial_url,
            api_key,
            self._in_memory_cache,
        )

    async def get_user_info(self) -> UserInfo:
        """Return user information."""
        request_url = urljoin(self._dial_url, "/v1/user/info")
        async with self._session.get(request_url, headers=self._headers) as response:
            try:
                response.raise_for_status()
            except ClientResponseError:
                logger.error(await response.text())
                raise

            data = await response.text()
            return UserInfo(**json.loads(data))

    async def get_application_info(self, application_id: str) -> dict[str, Any]:
        """
        Return extended information about an Application by its id.

        :param application_id: the id of DIAL application
        """
        request_url = urljoin(self._dial_url, f"/openai/applications/{application_id}")
        async with self._session.get(request_url, headers=self._headers) as response:
            try:
                response.raise_for_status()
            except ClientResponseError as e:
                logger.error(f"{str(e)}, {await response.text()}")
                raise e

            return await response.json()

    def get_file_storage(self) -> FileStorage:
        """Return the file storage object."""
        file_storage = DialFileStorage(
            self._session,
            self._dial_url,
            self._api_key,
        )
        if self._in_memory_cache is not None:
            return CachingFileStorage(file_storage, self._in_memory_cache)
        return file_storage

    def get_embeddings_model(self, deployment: str, max_retries=3) -> AzureOpenAIEmbeddings:
        """
        Return embedding model with given deployment name.

        :param deployment: a name of model deployment
        :param max_retries: maximum number of retries to make when performing requests to the model
        """
        # noinspection PyTypeChecker
        return AzureOpenAIEmbeddings(
            azure_endpoint=self._dial_url,
            api_key=self._api_key,
            api_version=OPENAI_API_VERSION,
            deployment=deployment,
            check_embedding_ctx_length=False,
            model_kwargs={"encoding_format": "float"},
            max_retries=max_retries,
            chunk_size=512,
            timeout=httpx.Timeout(60.0, connect=5.0),
        )

    def get_llm(self, config: LlmConfig) -> AzureChatOpenAI:
        """
        Return LLM for given configuration.

        :param config: the LLM configuration
        """
        # noinspection PyTypeChecker
        return AzureChatOpenAI(
            azure_endpoint=self._dial_url,
            api_key=self._api_key,
            model=config.deployment_name,
            api_version=OPENAI_API_VERSION,
            openai_api_type="azure",
            temperature=config.temperature,
            streaming=True,
            max_retries=config.max_retries,
            callbacks=[LCMessageLogger()],
            timeout=httpx.Timeout(60.0, connect=5.0),
        )
