"""Cloud ingest + analytics API (FastAPI).

Edge workers POST event batches; dashboards and the ops team read aggregates.
Run with `rva serve-api` or `uvicorn retail_vision.cloud.api:app`.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from retail_vision.cloud.analytics import EventStore


class EventIn(BaseModel):
    event_type: str
    store_id: str
    camera_id: str
    timestamp: float
    global_id: str
    role: str
    zone: str | None = None
    employee_id: str | None = None
    confidence: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class EventBatch(BaseModel):
    events: list[EventIn] = Field(min_length=1, max_length=5000)


def create_app(db_path: str | None = None) -> FastAPI:
    store = EventStore(db_path or os.environ.get("RVA_DB_PATH", ":memory:"))
    camera_health: dict[str, dict] = {}

    app = FastAPI(title="Retail Vision Analytics API", version="0.1.0")
    app.state.store = store

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/v1/events", status_code=202)
    def ingest(batch: EventBatch) -> dict:
        n = store.insert(e.model_dump() for e in batch.events)
        return {"accepted": n}

    @app.post("/v1/cameras/{camera_id}/heartbeat", status_code=204)
    def heartbeat(camera_id: str, metrics: dict[str, Any]) -> None:
        camera_health[camera_id] = metrics

    @app.get("/v1/cameras/health")
    def cameras_health() -> dict[str, dict]:
        return camera_health

    @app.get("/v1/analytics/summary")
    def summary(store_id: str | None = Query(default=None)) -> dict:
        return store.summary(store_id)

    @app.get("/v1/analytics/visits")
    def visits() -> list[dict]:
        return store.store_visits()

    @app.get("/v1/analytics/dwell")
    def dwell(store_id: str | None = Query(default=None)) -> list[dict]:
        return store.zone_dwell(store_id)

    @app.get("/v1/analytics/shelves")
    def shelves() -> list[dict]:
        return store.shelf_engagement()

    @app.get("/v1/analytics/employees")
    def employees() -> list[dict]:
        return store.employee_presence()

    @app.get("/v1/analytics/journeys")
    def journeys(limit: int = Query(default=20, le=500)) -> list[dict]:
        return store.cross_camera_journeys(limit)

    @app.get("/v1/review-queue")
    def review_queue(limit: int = Query(default=50, le=1000)) -> list[dict]:
        rows = store.review_queue(limit)
        if rows is None:
            raise HTTPException(500, "query failed")
        return rows

    return app


app = create_app()
