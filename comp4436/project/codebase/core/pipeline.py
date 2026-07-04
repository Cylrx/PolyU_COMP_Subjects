from __future__ import annotations

from typing import TYPE_CHECKING

from config import SimulationConfig
from core.cloud import CloudNode
from core.edge import EdgeNode
from core.events import EventKind
from core.queue import ProcessingQueue
from core.simulator import Simulator
from core.types import DataSample, InferenceRequest, Location, ModelVariant, ProfileCache
from evaluation.metrics import MetricsCollector, MetricsSummary
from strategies.protocols import AdmissionResult, DispatchAction, DispatchContext

if TYPE_CHECKING:
    from strategies.protocols import (
        AdmissionStrategy,
        DispatchStrategy,
    )


class Pipeline:
    """Wires all components together and orchestrates the simulation.

    All strategies are injected via the constructor.  The pipeline owns the
    Simulator and registers event handlers that coordinate data flow.
    """

    def __init__(
        self,
        profile: ProfileCache,
        queue: ProcessingQueue,
        edge: EdgeNode,
        cloud: CloudNode,
        admission: AdmissionStrategy,
        dispatcher: DispatchStrategy,
        arrival_times: tuple[float, ...],
        available_models: list[ModelVariant],
        metrics: MetricsCollector,
        config: SimulationConfig,
        metric_interval: float = 5.0,
    ) -> None:
        self._profile = profile
        self._queue = queue
        self._edge = edge
        self._cloud = cloud
        self._admission = admission
        self._dispatcher = dispatcher
        self._arrival_times = arrival_times
        self._available_models = available_models
        self._metrics = metrics
        self._config = config
        self._metric_interval = metric_interval

        self._simulator = Simulator()
        self._simulator.register(EventKind.SAMPLE_ARRIVAL, self._on_sample_arrival)
        self._simulator.register(
            EventKind.INFERENCE_COMPLETE, self._on_inference_complete
        )
        self._simulator.register(
            EventKind.MODEL_LOAD_COMPLETE, self._on_model_load_complete
        )

    def run(self) -> MetricsSummary:
        """Generate arrivals, run the simulation, return metrics summary."""
        arrival_times = self._arrival_times
        profile_size = self._profile.size

        for i, arrival_time in enumerate(arrival_times):
            dataset_idx = i % profile_size
            label = self._profile.lookup(dataset_idx, self._available_models[0].name).label
            sample = DataSample(
                id=i,
                dataset_idx=dataset_idx,
                label=label,
                arrival_time=arrival_time,
                deadline=arrival_time + self._config.deadline_budget,
            )
            self._simulator.schedule(arrival_time, EventKind.SAMPLE_ARRIVAL, sample)

        self._simulator.run()
        return self._metrics.summary(
            duration=self._config.duration,
            metric_interval=self._metric_interval,
        )

    # -- Event handlers -------------------------------------------------------

    def _on_sample_arrival(self, event: object) -> None:
        from core.events import Event

        assert isinstance(event, Event)
        sample: DataSample = event.payload

        self._metrics.record_arrival(sample.id, event.time)

        # Already expired on arrival?
        if sample.deadline <= self._simulator.clock:
            self._metrics.record_drop(sample.id, event.time, "expired_on_arrival")
            return

        result = self._admission.evaluate(
            sample, self._queue, self._simulator.clock
        )

        if result is AdmissionResult.REJECT:
            self._metrics.record_drop(sample.id, event.time, "admission_reject")
            return

        if result is AdmissionResult.ADMIT_EVICT_OLDEST:
            if self._queue.is_full() and not self._queue.is_empty():
                evicted = self._queue.pop_oldest()
                self._metrics.record_drop(
                    evicted.id, event.time, "evicted_by_admission"
                )

        if self._queue.is_full():
            self._metrics.record_drop(sample.id, event.time, "queue_full")
            return

        self._queue.enqueue(sample)
        self._metrics.record_enqueue(sample.id, event.time, self._queue.size)

        self._try_dispatch()

    def _on_inference_complete(self, event: object) -> None:
        from core.events import Event

        assert isinstance(event, Event)
        request: InferenceRequest = event.payload

        # Free the resource
        if request.location is Location.EDGE:
            self._edge.release()
        else:
            self._cloud.finish_request(event.time)

        # Look up profiled result
        profile_entry = self._profile.lookup(
            request.sample.dataset_idx, request.model_name
        )

        self._metrics.record_inference(
            sample_id=request.sample.id,
            prediction=profile_entry.prediction,
            correct=profile_entry.correct,
            model_name=request.model_name,
            location=request.location,
            arrival_time=request.sample.arrival_time,
            deadline=request.sample.deadline,
            start_time=request.start_time,
            end_time=event.time,
        )

        # Try to process next queued sample
        self._try_dispatch()

    def _on_model_load_complete(self, event: object) -> None:
        from core.events import Event

        assert isinstance(event, Event)
        model_name: str
        request: InferenceRequest
        model_name, request = event.payload

        # Record model switch
        self._metrics.record_model_switch(
            time=event.time,
            from_model=self._edge.loaded_model,
            to_model=model_name,
            load_duration=self._edge.model_load_time,
        )
        self._edge.loaded_model = model_name

        # Now start the actual edge inference
        self._start_edge_inference(request)

    # -- Dispatch logic -------------------------------------------------------

    def _try_dispatch(self) -> None:
        """Try to start inference for queued samples on available resources."""
        while not self._queue.is_empty():
            edge_avail = self._edge.is_available()
            cloud_avail = self._cloud.is_available()

            if not edge_avail and not cloud_avail:
                break

            sample = self._queue.peek()

            # Drop expired samples sitting in queue
            if sample.deadline <= self._simulator.clock:
                self._queue.dequeue()
                self._metrics.record_drop(
                    sample.id, self._simulator.clock, "expired_in_queue"
                )
                continue

            context = DispatchContext(
                current_time=self._simulator.clock,
                queue_capacity=self._queue.capacity,
                queued_samples=self._queue.snapshot(),
                available_models=tuple(self._available_models),
                edge_available=edge_avail,
                edge_next_available_time=(
                    self._simulator.clock
                    if edge_avail
                    else self._edge.next_available_time
                ),
                edge_loaded_model=self._edge.loaded_model,
                edge_model_load_time=self._edge.model_load_time,
                cloud_available=cloud_avail,
                cloud_next_available_time=(
                    self._simulator.clock
                    if cloud_avail
                    else self._cloud.next_available_time
                ),
                expected_cloud_rtt=self._cloud.network.base_rtt,
            )
            decision = self._dispatcher.decide(sample, context)

            if decision.action is DispatchAction.WAIT:
                break

            sample = self._queue.dequeue()

            if decision.action is DispatchAction.DROP:
                self._metrics.record_drop(
                    sample.id,
                    self._simulator.clock,
                    "dispatch_drop",
                )
                continue

            assert decision.action is DispatchAction.DISPATCH
            assert decision.location is not None
            assert decision.model_name is not None

            request = InferenceRequest(
                sample=sample,
                model_name=decision.model_name,
                location=decision.location,
                start_time=self._simulator.clock,
            )

            if decision.location is Location.EDGE:
                if self._edge.needs_model_switch(decision.model_name):
                    # Schedule model load, then inference starts after
                    load_complete_time = self._simulator.clock + self._edge.model_load_time
                    self._edge.reserve_until(load_complete_time)
                    self._simulator.schedule(
                        load_complete_time,
                        EventKind.MODEL_LOAD_COMPLETE,
                        (decision.model_name, request),
                    )
                else:
                    self._start_edge_inference(request)
            else:
                self._start_cloud_inference(request)

    def _start_edge_inference(self, request: InferenceRequest) -> None:
        profile_entry = self._profile.lookup(
            request.sample.dataset_idx, request.model_name
        )
        latency = self._edge.compute_latency(profile_entry.edge_latency)
        end_time = self._simulator.clock + latency
        self._edge.reserve_until(end_time)
        self._simulator.schedule(
            end_time,
            EventKind.INFERENCE_COMPLETE,
            request,
        )

    def _start_cloud_inference(self, request: InferenceRequest) -> None:
        profile_entry = self._profile.lookup(
            request.sample.dataset_idx, request.model_name
        )
        latency = self._cloud.compute_latency(profile_entry.cloud_latency)
        end_time = self._simulator.clock + latency
        self._cloud.start_request(end_time)
        self._simulator.schedule(
            end_time,
            EventKind.INFERENCE_COMPLETE,
            request,
        )
