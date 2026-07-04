from __future__ import annotations


class EdgeNode:
    """Simulates an edge device with limited compute and memory.

    Can hold exactly one model at a time. Switching models incurs a
    loading delay.
    """

    def __init__(
        self,
        model_load_time: float = 0.05,
    ) -> None:
        self.model_load_time = model_load_time
        self.loaded_model: str | None = None
        self.busy: bool = False
        self.next_available_time: float = 0.0

    def is_available(self) -> bool:
        return not self.busy

    def needs_model_switch(self, model_name: str) -> bool:
        return self.loaded_model != model_name

    def reserve_until(self, end_time: float) -> None:
        self.busy = True
        self.next_available_time = end_time

    def release(self) -> None:
        self.busy = False

    def compute_latency(self, profiled_latency: float) -> float:
        """Edge inference time is read directly from the edge profile."""
        return profiled_latency
