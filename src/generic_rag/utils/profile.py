import functools
import inspect
import logging
import time
from contextlib import asynccontextmanager


def log_execution_time(logger: logging.Logger):
    """Decorator to measure and log execution time of async function"""

    @asynccontextmanager
    async def _measure_execution_time(target):
        started = time.perf_counter()
        exc = None
        try:
            yield
        except BaseException as e:
            exc = e
            raise
        finally:
            elapsed_time = round(time.perf_counter() - started, 2)
            if exc is None:
                logger.debug(f"'{target.__qualname__}': execution took {elapsed_time:.2f} second(s)")
            else:
                logger.warning(
                    f"'{target.__qualname__}': completed with error after {elapsed_time:.2f} second(s)"
                )

    def decorator(target):
        if inspect.isasyncgenfunction(target):

            @functools.wraps(target)
            async def wrapper(*args, **kwargs):
                async with _measure_execution_time(target):
                    async for item in target(*args, **kwargs):
                        yield item

            return wrapper

        if inspect.iscoroutinefunction(target):

            @functools.wraps(target)
            async def wrapper(*args, **kwargs):
                async with _measure_execution_time(target):
                    return await target(*args, **kwargs)

            return wrapper

        raise RuntimeError(f"unsupported target: {repr(target)}")

    return decorator
