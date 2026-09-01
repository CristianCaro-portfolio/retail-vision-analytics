"""End-to-end: synthetic store -> detector -> tracker -> ReID -> events -> DuckDB analytics."""

import json

from fastapi.testclient import TestClient

from retail_vision.cloud.analytics import EventStore
from retail_vision.cloud.api import create_app
from retail_vision.detection.mock import ColorBlobDetector
from retail_vision.detection.oracle import OracleDetector
from retail_vision.pipeline.runner import SimulationRunner, StoreRuntime
from retail_vision.reid.embedder import build_embedder
from retail_vision.reid.gallery import Gallery
from retail_vision.simulation.evaluate import evaluate_identities
from retail_vision.simulation.world import build_demo_world
from retail_vision.sinks import JsonlSink, MemorySink


def build_runtime(cfg, world, sink, detector="oracle"):
    embedder = build_embedder(cfg.reid)
    gallery = Gallery(embedder.dim)
    for actor in world.actors:
        if actor.is_employee:
            gallery.add(actor.name, embedder.embed(world.render_enrolment_crop(actor)))
    runtime = StoreRuntime(cfg, sink=sink, employee_gallery=gallery)
    for i, (cid, w) in enumerate(runtime.workers.items()):
        if detector == "oracle":
            w.detector = OracleDetector(world, cid, seed=i)
        else:
            w.detector = ColorBlobDetector(background=world.render_background(cid), seed=i)
    return runtime


def test_identity_quality_on_synthetic_store(store_config):
    world = build_demo_world()
    runtime = build_runtime(store_config, world, MemorySink())
    report = evaluate_identities(runtime, world, frames=400)
    assert report.role_accuracy >= 0.95
    assert report.employee_id_accuracy >= 0.95
    assert report.purity >= 0.95
    assert report.fragmentation["emp-001"] == 1
    assert report.fragmentation["emp-002"] == 1


def test_pixel_only_blob_detector_still_identifies_employees(store_config):
    world = build_demo_world()
    runtime = build_runtime(store_config, world, MemorySink(), detector="blob")
    report = evaluate_identities(runtime, world, frames=200)
    assert report.employee_id_accuracy >= 0.9
    assert report.role_accuracy >= 0.95


def test_events_flow_to_analytics_and_api(store_config, tmp_path):
    world = build_demo_world()
    path = tmp_path / "events.jsonl"
    runtime = build_runtime(store_config, world, JsonlSink(path))
    events = SimulationRunner(runtime, world).run(400)
    runtime.close()
    assert events, "pipeline produced no events"
    types = {e.event_type for e in events}
    assert {"person_seen", "zone_entered", "zone_exited", "dwell", "product_interaction"} <= types

    store = EventStore()
    assert store.load_jsonl(path) == len(events)
    summary = store.summary()
    assert summary["employees_seen"] == 2
    assert summary["unique_customers"] >= 3
    journeys = store.cross_camera_journeys()
    assert any(j["global_id"] == "emp-001" and j["cameras"] == 2 for j in journeys)
    shelves = store.shelf_engagement()
    assert {s["zone"] for s in shelves} == {"shelf-A", "shelf-B"}

    client = TestClient(create_app(":memory:"))
    payload = {"events": [json.loads(line) for line in path.read_text().splitlines()]}
    r = client.post("/v1/events", json=payload)
    assert r.status_code == 202 and r.json()["accepted"] == len(events)
    assert client.get("/v1/analytics/summary").json()["events"] == len(events)
    assert client.get("/v1/analytics/employees").json()[0]["employee_id"] == "emp-001"
    assert client.get("/v1/review-queue").status_code == 200


def test_camera_metrics_snapshot(store_config):
    world = build_demo_world()
    runtime = build_runtime(store_config, world, MemorySink())
    SimulationRunner(runtime, world).run(120)
    snap = runtime.metrics()["cam-01"]
    assert snap["frames"] == 120
    assert snap["latency_ms_p95"] >= snap["latency_ms_p50"] > 0
    assert 0.0 <= snap["review_ratio"] <= 1.0
