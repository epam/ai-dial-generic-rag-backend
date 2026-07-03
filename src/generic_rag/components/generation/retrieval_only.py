from generic_rag.types import AnswerCallback, AnswerGenerator, Retriever


class RetrievalOnlyAnswerGenerator(AnswerGenerator):
    """Returns all retrieval results as attachments without LLM invocation."""

    async def invoke(self, query: str, retriever: Retriever, callback: AnswerCallback):
        """
        Generate answer to given user's query.

        :param query: the user query to answer
        :param retriever: the :class:`Retriever` used to find relevant chunk information
        :param callback: a callback to catch answer as it is generated
        """
        for i, doc in enumerate(await retriever.invoke(query)):
            callback.append_reference(i, doc)
