import asyncio
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from itertools import chain


async def rank_fusion[T](
    doc_lists: Sequence[Sequence[T]],
    key: Callable[[T], Hashable],
    weights: Sequence[float] | None = None,
    c: int = 60,
) -> Sequence[T]:
    """Perform weighted Reciprocal Rank Fusion on multiple rank lists.

    You can find more details about RRF here:
    https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf.

    :param doc_lists: a list of rank lists, where each rank list contains unique items.
    :param key: a function returning the value used to determine unique documents.
    :param weights: a list of weights corresponding to the lists (defaults to equal weighting).
    :param c: a constant added to the rank, controlling the balance between the importance
        of high-ranked items and the consideration given to lower-ranked items.
    """

    def _calculate() -> Sequence[T]:
        nonlocal weights

        if not weights:
            weights = [1 / len(doc_lists)] * len(doc_lists)
        if len(doc_lists) != len(weights):
            raise ValueError("Number of rank lists must be equal to the number of weights.")

        # associate each doc's content with its RRF score for later sorting by it
        # duplicated contents across retrievers are collapsed & scored cumulatively
        rrf_score: dict[Hashable, float] = defaultdict(float)
        for doc_list, weight in zip(doc_lists, weights, strict=True):
            for rank, doc in enumerate(doc_list, start=1):
                rrf_score[key(doc)] += weight / (rank + c)

        # docs are deduplicated by key function and then sorted by their scores
        all_docs = chain.from_iterable(doc_lists)
        return sorted(
            _unique_by_key(all_docs, key),
            reverse=True,
            key=lambda _doc: rrf_score[key(_doc)],
        )

    return await asyncio.to_thread(_calculate)


def _unique_by_key[T](iterable: Iterable[T], key: Callable[[T], Hashable]) -> Iterator[T]:
    """Yield unique elements of an iterable based on a key function."""
    seen = set()
    for e in iterable:
        if (k := key(e)) not in seen:
            seen.add(k)
            yield e
