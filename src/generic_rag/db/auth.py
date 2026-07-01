import asyncio
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime

from azure.identity.aio import ManagedIdentityCredential

DB_MSI_SCOPE = os.getenv("DB_MSI_SCOPE", "https://ossrdbms-aad.database.windows.net/.default")

logger = logging.getLogger(__name__)


class TokenProvider(ABC):
    @property
    @abstractmethod
    def token(self) -> str | None:
        """Return value of access token."""


class MsiTokenProvider(TokenProvider):
    _token: str | None = None
    _initialized = asyncio.Event()
    _refresh_task = None

    async def __aenter__(self):
        self._refresh_task = asyncio.create_task(self._refresh_token())
        await self._initialized.wait()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._refresh_task.cancel()

    @property
    def token(self) -> str | None:
        return self._token

    async def _refresh_token(self):
        try:
            while True:
                credential = ManagedIdentityCredential()
                access_token = await credential.get_token(DB_MSI_SCOPE)

                self._token = access_token.token
                self._initialized.set()

                token_expires_on = datetime.fromtimestamp(access_token.expires_on)
                next_refresh_delay = (token_expires_on - datetime.now()).total_seconds() - 120

                logger.info(f"token will expire on: {token_expires_on}")
                await asyncio.sleep(next_refresh_delay)

        except asyncio.CancelledError:
            pass
