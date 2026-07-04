from core.queue import ProcessingQueue
from core.types import DataSample

import pytest


def _sample(id: int = 0, deadline: float = 10.0) -> DataSample:
    return DataSample(
        id=id, dataset_idx=id, label=0, arrival_time=0.0, deadline=deadline
    )


class TestProcessingQueue:
    def test_enqueue_dequeue_fifo(self) -> None:
        q = ProcessingQueue(capacity=3)
        q.enqueue(_sample(id=1))
        q.enqueue(_sample(id=2))
        q.enqueue(_sample(id=3))

        assert q.dequeue().id == 1
        assert q.dequeue().id == 2
        assert q.dequeue().id == 3

    def test_is_full_at_capacity(self) -> None:
        q = ProcessingQueue(capacity=2)
        assert not q.is_full()
        q.enqueue(_sample(id=0))
        assert not q.is_full()
        q.enqueue(_sample(id=1))
        assert q.is_full()

    def test_enqueue_when_full_raises(self) -> None:
        q = ProcessingQueue(capacity=1)
        q.enqueue(_sample(id=0))

        with pytest.raises(OverflowError):
            q.enqueue(_sample(id=1))

    def test_dequeue_when_empty_raises(self) -> None:
        q = ProcessingQueue(capacity=5)

        with pytest.raises(IndexError):
            q.dequeue()

    def test_peek_returns_front_without_removing(self) -> None:
        q = ProcessingQueue(capacity=3)
        q.enqueue(_sample(id=1))
        q.enqueue(_sample(id=2))

        assert q.peek().id == 1
        assert q.size == 2

    def test_peek_when_empty_raises(self) -> None:
        q = ProcessingQueue(capacity=3)

        with pytest.raises(IndexError):
            q.peek()

    def test_pop_oldest_same_as_dequeue(self) -> None:
        q = ProcessingQueue(capacity=3)
        q.enqueue(_sample(id=1))
        q.enqueue(_sample(id=2))

        assert q.pop_oldest().id == 1

    def test_remove_expired(self) -> None:
        q = ProcessingQueue(capacity=5)
        q.enqueue(_sample(id=0, deadline=0.5))
        q.enqueue(_sample(id=1, deadline=1.5))
        q.enqueue(_sample(id=2, deadline=0.3))
        q.enqueue(_sample(id=3, deadline=2.0))

        expired = q.remove_expired(current_time=1.0)

        assert [s.id for s in expired] == [0, 2]
        assert q.size == 2
        assert q.dequeue().id == 1
        assert q.dequeue().id == 3

    def test_size_property(self) -> None:
        q = ProcessingQueue(capacity=5)
        assert q.size == 0
        q.enqueue(_sample(id=0))
        assert q.size == 1
        q.dequeue()
        assert q.size == 0

    def test_invalid_capacity_raises(self) -> None:
        with pytest.raises(ValueError):
            ProcessingQueue(capacity=0)
        with pytest.raises(ValueError):
            ProcessingQueue(capacity=-1)
