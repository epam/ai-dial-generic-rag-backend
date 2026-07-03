import logging
from functools import cache

from aidial_sdk.embeddings import Embedding, Embeddings, Request, Response, Usage
from langchain_community.embeddings.huggingface import DEFAULT_QUERY_BGE_INSTRUCTION_EN
from langchain_huggingface import HuggingFaceEmbeddings

from generic_rag.utils.profile import log_execution_time

logger = logging.getLogger(__name__)

BUILTIN_EMBEDDING_MODEL_NAME = "epam/bge-small-en"


@cache
def _get_builtin_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name=BUILTIN_EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
        },
        query_encode_kwargs={
            "normalize_embeddings": True,
            "prompt": DEFAULT_QUERY_BGE_INSTRUCTION_EN,
        },
        show_progress=False,
    )


class EmbeddingsEndpoint(Embeddings):
    def __init__(self):
        _get_builtin_embeddings_model()  # load model

    @log_execution_time(logger)
    async def embeddings(self, request: Request) -> Response:
        model = _get_builtin_embeddings_model()
        data = [
            Embedding(
                embedding=embedding,
                index=i,
            )
            for i, embedding in enumerate(await model.aembed_documents(request.input))
        ]
        return Response(
            data=data,
            model=request.model,
            usage=Usage(
                prompt_tokens=0,
                total_tokens=0,
            ),
        )
