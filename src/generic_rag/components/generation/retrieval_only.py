from generic_rag.types import Answer, AnswerGenerator, Retriever


class RetrievalOnlyAnswerGenerator(AnswerGenerator):
    """Returns all retrieval results as attachments without LLM invocation."""

    async def invoke(self, query: str, retriever: Retriever, answer: Answer):
        """
        Generate answer to given user's query.

        :param query: the user query to answer
        :param retriever: the :class:`Retriever` used to find relevant chunk information
        :param answer: the current answer
        """
        for doc in await retriever.invoke(query, answer):
            await answer.add_citation(doc)
