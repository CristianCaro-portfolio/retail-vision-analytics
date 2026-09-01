"""Store-level event de-duplication.

Twenty cameras often see the same aisle. Once identities are global, the same
person entering the same logical zone can be reported by two cameras a few
frames apart. The deduplicator suppresses repeats of (identity, event, zone)
within a time window, so the cloud counts one entry, not two.
"""

from __future__ import annotations

from retail_vision.types import Event


class EventDeduplicator:
    def __init__(self, window_seconds: float = 10.0) -> None:
        self.window = window_seconds
        self._last: dict[tuple[str, str, str | None], float] = {}

    def accept(self, event: Event) -> bool:
        key = (event.global_id, event.event_type, event.zone)
        last = self._last.get(key)
        if last is not None and event.timestamp - last < self.window:
            return False
        self._last[key] = event.timestamp
        return True

    def prune(self, now: float) -> None:
        self._last = {k: t for k, t in self._last.items() if now - t < self.window}
