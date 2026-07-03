from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Self

from annotated_types import Interval
from pydantic import BaseModel


@dataclass(frozen=True)
class Pagination:
    offset: Annotated[int, Interval(ge=0)]
    limit: Annotated[int, Interval(ge=0)]


class PaginatedResults[T](BaseModel):
    total_count: Annotated[int, Interval(ge=0)]
    offset: Annotated[int, Interval(ge=0)]
    limit: Annotated[int, Interval(ge=0)]
    results: Sequence[T]

    @classmethod
    def create(cls, results: Sequence[T], pagination: Pagination, total_count: int) -> Self:
        return cls(
            results=results,
            offset=pagination.offset,
            limit=pagination.limit,
            total_count=total_count,
        )
