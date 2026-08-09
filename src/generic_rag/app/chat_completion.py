import json
import logging

from aidial_sdk.chat_completion import ChatCompletion, Request, Response
from aidial_sdk.deployment.configuration import ConfigurationRequest, ConfigurationResponse
from aidial_sdk.exceptions import InternalServerError
from langchain_community.callbacks import get_openai_callback
from pydantic import SecretStr, ValidationError
from pydantic_partial import create_partial_model

from generic_rag.channel import Channel, RequestConfig
from generic_rag.scope import ChannelBindings
from generic_rag.types import AbstractAnswer, AnswerGenerator, Retriever
from generic_rag.utils.answers import DialAnswer, SharingManager

logger = logging.getLogger(__name__)


class ChannelCompletion(ChatCompletion):
    _enable_debug_stages: bool = True

    async def chat_completion(self, request: Request, response: Response) -> None:
        """Chat completion entrypoint."""
        async with ChannelBindings(SecretStr(request.api_key), request.dial_application_id).scope.adefine():
            answer = SharingManager(
                DialAnswer(response.create_single_choice(), enable_debug_stages=self._enable_debug_stages)
            )
            with answer:
                try:
                    with answer.create_stage("Channel configuration", debug=True) as stage:
                        channel = await Channel.get_current_channel()

                        stage.append_content(f"application id: `{request.dial_application_id}`\n\n")
                        stage.append_content(
                            f"```json\n{json.dumps(channel.dump_config(), indent=2)}\n```\n\n"
                        )

                    await self._channel_completion(answer, request, response, channel)

                except Exception as e:
                    with answer.create_stage("Internal error", debug=True):
                        raise InternalServerError(f"Unexpected error: {str(e)}") from e

    @staticmethod
    async def _channel_completion(
        answer: AbstractAnswer, request: Request, response: Response, channel: Channel
    ):
        """Chat completion logic for specific channel."""
        with answer.create_stage("Request configuration", debug=True) as stage:
            request_config_model = await RequestConfig.get_dynamic_model()
            raw_request = request.custom_fields.configuration if request.custom_fields else None
            try:
                request_config = request_config_model.create(
                    defaults=channel.request_config,
                    overrides=raw_request,
                )
            except ValidationError as e:
                stage.append_content(f"```\n{e.__class__.__name__}: {str(e)}\n```\n\n")
                if isinstance(raw_request, dict):
                    stage.append_content(f"```json\n{json.dumps(raw_request, indent=2)}\n```\n\n")
                raise InternalServerError("Unable to process request configuration") from e
            else:
                stage.append_content(
                    f"```json\n{request_config.model_dump_json(indent=2, exclude_none=True)}\n```\n\n"
                )

        last_message = str(request.messages[-1].content)

        retriever = Retriever.create(request_config.retriever)
        answer_generator = AnswerGenerator.create(request_config.generation)

        with get_openai_callback() as cb:
            await answer_generator.invoke(last_message, retriever, answer)

            with answer.create_stage("Token usage", debug=True) as stage:
                stage.append_content(f"```text\n{str(cb)}\n```\n\n")

            response.set_usage(cb.prompt_tokens, cb.completion_tokens)

    async def configuration(self, request: ConfigurationRequest) -> ConfigurationResponse | dict:
        """Chat completion configuration entrypoint."""
        async with ChannelBindings(SecretStr(request.api_key), request.dial_application_id).scope.adefine():
            configuration_model = create_partial_model(await RequestConfig.get_dynamic_model())
            return configuration_model.model_json_schema()
