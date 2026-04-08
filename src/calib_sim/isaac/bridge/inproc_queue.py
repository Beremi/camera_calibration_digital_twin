"""Simple typed in-process queue."""

from __future__ import annotations

from collections import deque
from typing import Generic, TypeVar


T = TypeVar("T")


class InprocQueue(Generic[T]):
    def __init__(self) -> None:
        self._items: deque[T] = deque()

    def put(self, item: T) -> None:
        self._items.append(item)

    def get_all(self) -> list[T]:
        items = list(self._items)
        self._items.clear()
        return items

    def __len__(self) -> int:
        return len(self._items)
