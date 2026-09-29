"""Batching helper (``itertools.batched`` only exists from Python 3.12)."""

from collections.abc import Iterable, Iterator
from itertools import islice
from typing import TypeVar

T = TypeVar("T")


def batched(items: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield successive lists of at most ``size`` items."""
    if size < 1:
        raise ValueError("size must be at least 1")
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch
