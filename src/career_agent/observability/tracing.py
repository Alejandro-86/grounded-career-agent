"""Tracing behind a tiny interface, with an MLflow implementation.

Tracing must never be able to break a run, so the MLflow tracer degrades to a
no-op if the tracking backend is unavailable.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol

# Synchronous export so a trace is queryable the moment its root span closes.
os.environ.setdefault("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


class SpanHandle(Protocol):
    def set_outputs(self, outputs: Any) -> None: ...
    def set_attribute(self, key: str, value: Any) -> None: ...


class Tracer(Protocol):
    def span(self, name: str, *, span_type: str = "CHAIN", inputs: Any = None) -> Any: ...
    def last_trace_id(self) -> str | None: ...


class _NoopSpan:
    def set_outputs(self, outputs: Any) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass


class NoopTracer:
    @contextmanager
    def span(self, name: str, *, span_type: str = "CHAIN", inputs: Any = None) -> Iterator[Any]:
        yield _NoopSpan()

    def last_trace_id(self) -> str | None:
        return None


class MLflowTracer:
    def __init__(
        self,
        tracking_uri: str = "sqlite:///mlflow.db",
        experiment: str = "grounded-career-agent",
    ):
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self._mlflow = mlflow

    @contextmanager
    def span(self, name: str, *, span_type: str = "CHAIN", inputs: Any = None) -> Iterator[Any]:
        with self._mlflow.start_span(name=name, span_type=span_type) as sp:
            if inputs is not None:
                sp.set_inputs(inputs)
            yield sp

    def last_trace_id(self) -> str | None:
        return self._mlflow.get_last_active_trace_id()

    def log_eval(self, name: str, params: dict[str, Any], metrics: dict[str, float]) -> str:
        with self._mlflow.start_run(run_name=name) as run:
            self._mlflow.log_params({k: str(v) for k, v in params.items()})
            self._mlflow.log_metrics(metrics)
            return run.info.run_id


def default_tracer(enabled: bool = True) -> Tracer:
    if not enabled:
        return NoopTracer()
    try:
        return MLflowTracer(tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    except Exception:  # noqa: BLE001 - tracing must never take the agent down
        return NoopTracer()
