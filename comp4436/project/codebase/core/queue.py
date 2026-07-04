from __future__ import annotations

from collections import deque

from core.types import DataSample


class ProcessingQueue:
    """Bounded FIFO queue for data samples awaiting inference."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"Queue capacity must be positive, got {capacity}")
        self.capacity = capacity
        self._items: deque[DataSample] = deque()

    def enqueue(self, sample: DataSample) -> None:
        if self.is_full():
            raise OverflowError("Queue is full")
        self._items.append(sample)

    def dequeue(self) -> DataSample:
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self._items.popleft()

    def peek(self) -> DataSample:
        """Return the front item without removing it."""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self._items[0]

    def pop_oldest(self) -> DataSample:
        """Remove and return the oldest (front) item. Semantic alias for dequeue."""
        return self.dequeue()

    def remove_expired(self, current_time: float) -> list[DataSample]:
        """Remove all samples whose deadline has passed. Returns evicted samples."""
        expired: list[DataSample] = []
        remaining: deque[DataSample] = deque()
        for sample in self._items:
            if sample.deadline <= current_time:
                expired.append(sample)
            else:
                remaining.append(sample)
        self._items = remaining
        return expired

    def is_full(self) -> bool:
        return len(self._items) >= self.capacity

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def snapshot(self) -> tuple[DataSample, ...]:
        return tuple(self._items)

    @property
    def size(self) -> int:
        return len(self._items)
