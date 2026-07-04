"""Admission control strategies.

Decide whether to accept or reject incoming data samples based on
queue state. Grounded in LEC8 baselines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from strategies.protocols import AdmissionResult

if TYPE_CHECKING:
    from core.queue import ProcessingQueue
    from core.types import DataSample


class DropTail:
    """Reject new samples when the queue is full.

    The simplest baseline: once capacity is reached, all further arrivals
    are discarded until space opens up. Preserves FIFO ordering of
    existing queue contents.
    """

    def evaluate(
        self, sample: DataSample, queue: ProcessingQueue, current_time: float
    ) -> AdmissionResult:
        if queue.is_full():
            return AdmissionResult.REJECT
        return AdmissionResult.ADMIT


class DropOld:
    """When queue is full, evict the oldest sample to admit the new one.

    Based on LEC8 Drop-Old policy: prioritises keeping the most recent
    data, under the assumption that newer samples carry fresher
    information about the environment.
    """

    def evaluate(
        self, sample: DataSample, queue: ProcessingQueue, current_time: float
    ) -> AdmissionResult:
        if queue.is_full():
            return AdmissionResult.ADMIT_EVICT_OLDEST
        return AdmissionResult.ADMIT
