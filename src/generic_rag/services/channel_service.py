import hashlib
import logging

from injection import find_instance, singleton

from generic_rag.channel import Channel, ChannelConfig
from generic_rag.dial_client import DialClient

logger = logging.getLogger(__name__)


@singleton
class ChannelService:
    """ Channel configuration management service. """

    _configuration_schema_model: type[ChannelConfig] = None

    async def get_channel(self, dial_application_id: str) -> Channel:
        """
        Return the channel for given DIAL application.

        :param dial_application_id: the id of DIAL application that initiated the request
            (gets passed by DIAL via `x-dial-application-id` header of the request)
        """
        dial_client = find_instance(DialClient)
        application_info = await dial_client.get_application_info(dial_application_id)
        application_properties = application_info.get("application_properties", {})

        if not application_properties:
            raise ValueError("Application properties not found")

        model = await self.get_channel_config_model()
        channel_config = model.model_validate(application_properties)

        file_storage = dial_client.get_file_storage()
        bucket = await file_storage.get_bucket()
        if (stream := await file_storage.download_file(f"files/{bucket}/channel_key")) is not None:
            channel_key = b"".join([chunk async for chunk in stream]).decode().strip()
        else:
            channel_key = hashlib.sha1(dial_application_id.encode()).hexdigest()
            await file_storage.put_file(bucket, "channel_key", "text/plain", channel_key.encode())

        return Channel(channel_key, channel_config)

    async def get_channel_config_model(self) -> type[ChannelConfig]:
        """ Return dynamic model of channel configuration schema. """
        if not self._configuration_schema_model:
            self._configuration_schema_model = await ChannelConfig.get_dynamic_model()
        return self._configuration_schema_model
