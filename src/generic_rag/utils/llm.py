import logging
import re
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult

logger = logging.getLogger(__name__)


class LCMessageLogger(AsyncCallbackHandler):
    # NOTE: According to https://python.langchain.com/docs/modules/callbacks/async_callbacks
    # "If you are planning to use the async API,
    # it is recommended to use AsyncCallbackHandler to avoid blocking the runloop."
    #
    # For sync callback handler, subclass from 'BaseCallbackHandler'

    """
    Default LangChain logging (when using set_debug(True)) produces looooots of redundant logs.
    Here we define our custom langchain logger.
    """

    RE_B64_IMAGE_IN_HISTORY = re.compile(r"(data:image/(?:\w+);base64,)(.*?)(\'|\"|\n)")

    @staticmethod
    def langchain_msg_2_role_content(msg: BaseMessage):
        return {"role": msg.type, "content": msg.content}

    def __init__(self, log_raw_llm_response: bool = True, log_token_usage: bool = False):
        """
        log_token_usage: whether we should log the use of tokens or not
        """

        super().__init__()
        self._log_raw_llm_response = log_raw_llm_response
        self._log_token_usage = log_token_usage

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[BaseMessage]], **kwargs: Any
    ) -> Any:
        """Run when Chat Model starts running."""
        if len(messages) != 1:
            raise ValueError(f'expected "messages" to have len 1, got: {len(messages)}')

        if serialized["id"][-1] == "AzureChatOpenAI":
            try:
                model = serialized["kwargs"]["model_name"]  # deployment_name
            except Exception:
                model = "<failed to determine LLM>"
        else:
            model = "<failed to determine LLM>"

        msgs_list = list(map(self.langchain_msg_2_role_content, messages[0]))
        msgs_str = "\n".join(map(str, msgs_list))
        # remove base64 encoded image from calls to gpt-4-vision.
        msgs_str = self.RE_B64_IMAGE_IN_HISTORY.sub(r"\1<base64_image>\3", msgs_str)

        logger.info(f"call to {model} with {len(msgs_list)} messages:\n{msgs_str}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        generations = response.generations
        if len(generations) != 1:
            raise ValueError(f'expected "generations" to have len 1, got: {len(generations)}')
        if len(generations[0]) != 1:
            raise ValueError(f'expected "generations[0]" to have len 1, got: {len(generations[0])}')

        if self._log_raw_llm_response:
            gen: ChatGeneration = generations[0][0]
            ai_msg = gen.message
            logger.info(f'raw LLM response: "{ai_msg.content}"')

        if self._log_token_usage:
            llm_output = response.llm_output
            if llm_output:
                token_usage = llm_output.get("token_usage")
                logger.info(f"LLM usage (from LLM response): {token_usage}")
            else:
                logger.warning("failed to extract extract LLM usage from LLM response: 'llm_output' is None")
