import logging
from collections.abc import AsyncIterable
from typing import BinaryIO

import humanfriendly
from cachetools import Cache

from generic_rag.types import FileMetadata, FileStorage

logger = logging.getLogger(__name__)


class CachingFileStorage(FileStorage):
    """Implementation of FileStorage with in-memory caching."""

    def __init__(self, storage: FileStorage, cache: Cache):
        self._storage = storage
        self._cache = cache

    async def get_bucket(self) -> str:
        return await self._storage.get_bucket()

    async def put_file(
        self, bucket: str, filepath: str, content_type: str, content: bytes | BinaryIO
    ) -> FileMetadata:
        file_metadata = await self._storage.put_file(bucket, filepath, content_type, content)
        if isinstance(content, bytes):
            cache_key = (file_metadata.url, file_metadata.etag)
            self._add_to_cache(cache_key, content)
        return file_metadata

    async def get_file_metadata(self, url: str) -> FileMetadata | None:
        return await self._storage.get_file_metadata(url)

    async def download_file(self, url: str) -> AsyncIterable[bytes] | None:
        if metadata := await self._storage.get_file_metadata(url):
            cache_key = (url, metadata.etag)

            if cache_key not in self._cache and float(metadata.content_length) < self._cache.maxsize:
                content = b"".join([chunk async for chunk in await self._storage.download_file(url)])
                self._add_to_cache(cache_key, content)

            if (content := self._cache.get(cache_key)) is not None:

                async def _content_gen():
                    yield content

                return _content_gen()

            return await self._storage.download_file(url)

        return None

    async def copy_file_to_user(self, source_url: str, destination_name: str) -> str:
        return await self._storage.copy_file_to_user(source_url, destination_name)

    async def delete_file(self, url: str):
        await self._storage.delete_file(url)

    def _add_to_cache(self, cache_key: tuple[str, str], content: bytes):
        try:
            self._cache[cache_key] = content
        except ValueError:
            value_size = humanfriendly.format_size(len(content), binary=True)
            max_cache_size = humanfriendly.format_size(int(self._cache.maxsize), binary=True)
            logger.warning(
                f"{self.__class__.__qualname__}: could not add '{cache_key}' "
                + f"(value size: {value_size}, max cache size: {max_cache_size})"
            )
        else:
            cache_size = humanfriendly.format_size(self._cache.currsize, binary=True)
            logger.debug(f"{self.__class__.__qualname__}: added '{cache_key}' (cache size: {cache_size})")
