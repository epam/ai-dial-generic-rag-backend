import logging
from abc import ABC, abstractmethod
from typing import cast

from injection import inject
from pydantic import BaseModel, Field, create_model

from generic_rag.channel import Channel
from generic_rag.db.session import get_current_session, transaction
from generic_rag.services.document_matcher import DocumentMatcher, DocumentMatcherConfig
from generic_rag.types import AbstractAnswer, AnswerStage, ConfigurableComponent

logger = logging.getLogger(__name__)


class DocumentSelector[ConfigT: BaseModel = BaseModel](ConfigurableComponent[ConfigT], ABC):
    """Implements the logic of getting filters configuration to be used by :class:`DocumentSelector`."""

    async def get_document_subset(self, answer: AbstractAnswer) -> list[int] | None:
        """
        Get a subset of documents to use for retrieval.

        :param answer: the current answer (to report retrieval stages execution)
        :return: list of document IDs, or special marker `None` if all available documents should be used.
        """
        with answer.create_stage("Document selector") as stage:
            return await self._get_document_subset(stage)

    @abstractmethod
    async def _get_document_subset(self, stage: AnswerStage) -> list[int] | None: ...


class AllDocumentsDocumentSelector(DocumentSelector):
    """Always use all available documents."""

    async def _get_document_subset(self, stage: AnswerStage) -> list[int] | None:
        return None


class ExactDocumentsDocumentSelectorConfig(BaseModel):
    document_ids: list[int] = Field(min_length=1, description="IDs of documents to be used.")


class ExactDocumentsDocumentSelector(DocumentSelector[ExactDocumentsDocumentSelectorConfig]):
    """Restrict search to specific documents."""

    async def _get_document_subset(self, stage: AnswerStage) -> list[int] | None:
        return self.config.document_ids


class MetadataDocumentSelector[ConfigT: BaseModel = BaseModel](DocumentSelector[ConfigT], ABC):
    """Restrict search to documents whose metadata matches given criteria."""

    @inject
    def __init__(self, config: ConfigT, channel: Channel = NotImplemented):
        super().__init__(config)
        self._channel_key = channel.channel_key

    @transaction
    async def _get_document_subset(self, stage: AnswerStage) -> list[int] | None:
        matcher_config = await self._get_matcher_config()

        if (query := DocumentMatcher(self._channel_key, matcher_config).get_query()) is None:
            stage.append_content("Use all available documents.\n\n")
            return None

        stage.append_content(
            f"Provided configuration:\n```json\n{matcher_config.model_dump_json(indent=2, exclude_none=True)}\n```\n\n"
        )

        async with get_current_session() as session:
            cursor = await session.scalars(query)
            result = list(cursor.all())

        stage.append_content(f"Found {len(result)} matching document(s).\n\n")

        return result

    @abstractmethod
    async def _get_matcher_config(self) -> DocumentMatcherConfig: ...


class ExplicitDocumentSelectorConfig(BaseModel, ABC):
    @classmethod
    @inject
    async def get_dynamic_model(
        cls, channel: Channel | None = None
    ) -> type["ExplicitDocumentSelectorConfig"]:
        if channel is not None:
            # noinspection PyTypeChecker
            return create_model(
                cls.__name__,
                __doc__=cls.__doc__,
                __base__=(
                    cls,
                    await DocumentMatcherConfig.get_dynamic_model(),
                ),
            )
        return cls


class ExplicitDocumentSelector[ConfigT: ExplicitDocumentSelectorConfig = ExplicitDocumentSelectorConfig](
    MetadataDocumentSelector[ConfigT]
):
    """Restrict search to documents whose metadata matches given criteria, defined explicitly on each request."""

    async def _get_matcher_config(self) -> DocumentMatcherConfig:
        if isinstance(self.config, DocumentMatcherConfig):
            return cast(DocumentMatcherConfig, cast(BaseModel, self.config))
        return await DocumentMatcherConfig.get_default_value()
