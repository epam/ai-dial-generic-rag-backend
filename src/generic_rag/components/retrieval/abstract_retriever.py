from abc import ABC, abstractmethod
from typing import Annotated, NotRequired, Self, TypedDict, cast

from annotated_types import MinLen
from injection import inject
from langchain_core.documents import Document as LangchainDocument
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field, NonNegativeInt, TypeAdapter, create_model

from generic_rag.channel import Channel
from generic_rag.components.retrieval.document_selector import (
    AllDocumentsDocumentSelector,
    DocumentSelector,
)
from generic_rag.components.retrieval.retrieval_stage import RetrievalStage
from generic_rag.components.search_index import ChunkIndex
from generic_rag.services.chunk_sources_manager import ChunkSourcesManager
from generic_rag.types import RetrievalStageListener, Retriever


class AbstractRetrieverRequest(BaseModel, ABC):
    """ Request-specific options of :class:`AbstractRetriever`. """
    top_k: dict[str, int | None]

    @classmethod
    async def get_dynamic_model(cls, channel: Channel) -> type["AbstractRetrieverRequest"]:
        # noinspection PyTypedDict
        top_k_fields = {
            idx.index_name: Annotated[
                NotRequired[NonNegativeInt],
                Field(description=f"Maximum number of results to be returned by `{idx.index_name}` index.")
            ]
            for idx in await channel.get_indexes()
        }
        # noinspection PyTypedDict
        top_k_model = TypedDict(AbstractRetrieverRequest.__name__ + "TopK", top_k_fields)
        top_k_default = {idx.index_name: idx.default_limit for idx in await channel.get_indexes()}

        return create_model(
            cls.__name__,
            __base__=cls,
            __doc__=cls.__doc__,
            top_k=Annotated[
                top_k_model,
                MinLen(1),
                Field(
                    top_k_default,
                    description=(
                        "Overrides for maximum number of results to be returned by given index. For the index "
                        "to be enabled for search corresponding non-zero value should be explicitly defined."
                    ),
                )
            ],
        )

    @classmethod
    async def get_default_value(cls, channel: Channel) -> Self:
        model = await cls.get_dynamic_model(channel)
        return model.model_validate({})


class AbstractRetrieverConfig(BaseModel, ABC):
    """ :class:`AbstractRetriever` configuration model. """
    document_selector: BaseModel

    @classmethod
    @inject
    async def get_dynamic_model(cls, channel: Channel = None) -> type["AbstractRetrieverConfig"]:
        document_selector_model = await DocumentSelector.get_aggregated_config_model()
        document_selector_default = TypeAdapter(document_selector_model).validate_python(
            {"type": AllDocumentsDocumentSelector.get_qualifier()}
        )

        base = cls if channel is None else (cls, await AbstractRetrieverRequest.get_dynamic_model(channel))

        return create_model(
            cls.__name__,
            __doc__=cls.__doc__,
            __base__=base,
            document_selector=Annotated[
                document_selector_model,
                Field(
                    document_selector_default,
                    description=(
                        "Allows to restrict search scope by selecting subset of documents."
                    )
                )
            ]
        )


class AbstractRetriever[ConfigT: AbstractRetrieverConfig = AbstractRetrieverConfig](Retriever[ConfigT], ABC):
    """ Base class for retrievers implementation. """

    @inject
    def __init__(self, config: ConfigT, channel: Channel, chunk_sources_manager: ChunkSourcesManager):
        super().__init__(config)
        self._channel = channel
        self._listener = RetrievalStageListener()
        self._chunk_sources_manager = chunk_sources_manager

    async def invoke(self, query: str) -> list[LangchainDocument]:
        """
        Retrieve pieces of information that are relevant to given query.

        :param query: the user query used for search
        """
        document_selector = DocumentSelector.create(
            self.config.document_selector
        ).use_listener(
            self._listener
        )
        documents = await document_selector.get_document_subset()

        if documents is not None and len(documents) < 1:
            return []

        request_config = cast(AbstractRetrieverRequest, self.config) \
            if isinstance(self.config, AbstractRetrieverRequest) \
            else await AbstractRetrieverRequest.get_default_value(self._channel)

        if intermediate_stages := [
            self._create_intermediate_stage(
                index,
                documents,
                request_config.top_k.get(index.index_name)
            )
            for index in await self._channel.get_indexes() if request_config.top_k.get(index.index_name)
        ]:
            final_stage = self._create_final_stage(intermediate_stages)
            return await final_stage.ainvoke(query)

        return []

    def use_listener(self, listener: RetrievalStageListener) -> Self:
        """ Use given retrieval event listener. """
        self._listener = listener
        return self

    @abstractmethod
    def _create_final_stage(self, intermediate_stages: list[RetrievalStage]) -> RetrievalStage:
        ...

    @abstractmethod
    def _create_intermediate_stage(self, index: ChunkIndex, documents: list[int] | None, top_k: int) -> RetrievalStage:
        ...

    def _stage_factory(self, stage_name: str, retriever: BaseRetriever) -> RetrievalStage:
        return RetrievalStage(
            stage_name=stage_name,
            retriever=retriever,
            listener=self._listener,
            sources_manager=self._chunk_sources_manager,
        )
