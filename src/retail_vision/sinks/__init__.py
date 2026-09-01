"""Event sinks: where the edge worker ships events.

Only events and (optionally) small crops leave the store. Raw video never does.
"""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import httpx

from retail_vision.config import SinkConfig
from retail_vision.types import Event


class EventSink(ABC):
    @abstractmethod
    def emit(self, event: Event) -> None: ...

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.flush()


class MemorySink(EventSink):
    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)


class StdoutSink(EventSink):
    def emit(self, event: Event) -> None:
        sys.stdout.write(json.dumps(event.to_dict()) + "\n")


class JsonlSink(EventSink):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")  # noqa: SIM115

    def emit(self, event: Event) -> None:
        self._fh.write(json.dumps(event.to_dict()) + "\n")

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class HttpBatchSink(EventSink):
    """POSTs batches to the cloud ingest API. Buffers locally if the network is down."""

    def __init__(self, url: str, batch_size: int = 50, timeout: float = 5.0) -> None:
        self.url = url
        self.batch_size = batch_size
        self._client = httpx.Client(timeout=timeout)
        self._buffer: list[dict] = []

    def emit(self, event: Event) -> None:
        self._buffer.append(event.to_dict())
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        try:
            resp = self._client.post(self.url, json={"events": self._buffer})
            resp.raise_for_status()
            self._buffer.clear()
        except httpx.HTTPError:
            # Keep buffering; a real deployment would spill to disk after a cap.
            if len(self._buffer) > self.batch_size * 100:
                del self._buffer[: self.batch_size]

    def close(self) -> None:
        self.flush()
        self._client.close()


def build_sink(cfg: SinkConfig) -> EventSink:
    if cfg.kind == "memory":
        return MemorySink()
    if cfg.kind == "stdout":
        return StdoutSink()
    if cfg.kind == "jsonl":
        return JsonlSink(cfg.path)
    if cfg.kind == "http":
        if not cfg.url:
            raise ValueError("sink.url is required for the http sink")
        return HttpBatchSink(cfg.url, cfg.batch_size)
    raise ValueError(f"unknown sink kind {cfg.kind!r}")
