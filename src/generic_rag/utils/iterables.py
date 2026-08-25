from collections.abc import AsyncIterable, Sequence


async def batched_async[T](iterable: AsyncIterable[T], batch_size: int) -> AsyncIterable[Sequence[T]]:
    """
    Async version of `itertools.batched` function.

    Reads data from given async iterable and return them as batches of given size.

    :param iterable: an AsyncIterable to read data from
    :param batch_size: required size of batches
    """
    batch: list[T] = []
    async for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
