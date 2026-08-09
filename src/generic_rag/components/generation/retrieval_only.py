from generic_rag.types import AbstractAnswer, AnswerGenerator, Retriever


class RetrievalOnlyAnswerGenerator(AnswerGenerator):
    """Returns all retrieval results as attachments without LLM invocation."""

    async def invoke(self, query: str, retriever: Retriever, answer: AbstractAnswer):
        """
        Generate answer to given user's query.

        :param query: the user query to answer
        :param retriever: the :class:`Retriever` used to find relevant chunk information
        :param answer: the current answer
        """
        for i, doc in enumerate(await retriever.invoke(query, answer), start=1):
            await answer.add_reference(i, doc)
