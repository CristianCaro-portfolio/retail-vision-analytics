"""Command line entry point: `rva --help`."""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(
    help="Retail Vision Analytics - edge pipeline, simulator, training and cloud tools."
)


def _echo(obj) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str))


@app.command()
def simulate(
    config: Path = typer.Option("configs/store_demo.yaml", help="Store config YAML."),
    frames: int = typer.Option(300, help="Frames to simulate per camera."),
    noise: float = typer.Option(0.0, help="Gaussian pixel noise std added to synthetic frames."),
    dropout: float = typer.Option(0.03, help="Probability that the detector misses a person."),
    detector: str = typer.Option("oracle", help="oracle (GT + noise) | blob (pixel-based mock)."),
    output: Path | None = typer.Option(None, help="Override the events JSONL path."),
    metrics_out: Path | None = typer.Option(None, help="Write per-camera metrics JSON here."),
    seed: int = 0,
) -> None:
    """Run the full pipeline over the synthetic two-camera store and write events."""
    from retail_vision.config import load_store_config
    from retail_vision.pipeline.runner import SimulationRunner
    from retail_vision.simulation.world import build_demo_world

    cfg = load_store_config(config)
    if output:
        cfg.sink.kind, cfg.sink.path = "jsonl", str(output)
    world = build_demo_world(seed=seed)
    world.noise_std = noise
    runtime = _runtime_with_demo_gallery(cfg, world, detector, dropout=dropout, seed=seed)
    events = SimulationRunner(runtime, world).run(frames)
    runtime.close()
    by_type: dict[str, int] = {}
    for e in events:
        by_type[e.event_type] = by_type.get(e.event_type, 0) + 1
    metrics = runtime.metrics()
    if metrics_out:
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(metrics, indent=2))
    _echo(
        {
            "frames": frames,
            "events": len(events),
            "events_by_type": by_type,
            "identities": runtime.resolver.active_identities(),
            "camera_metrics": metrics,
            "sink": cfg.sink.path if cfg.sink.kind == "jsonl" else cfg.sink.kind,
        }
    )


@app.command()
def evaluate(
    config: Path = typer.Option("configs/store_demo.yaml"),
    frames: int = 300,
    noise: float = 0.0,
    dropout: float = 0.03,
    detector: str = typer.Option("oracle", help="oracle (GT + noise) | blob (pixel-based mock)."),
    seed: int = 0,
    output: Path | None = typer.Option(None, help="Write the report JSON here."),
) -> None:
    """Score identity quality (fragmentation, purity, role and employee-id accuracy)
    against simulator ground truth."""
    from retail_vision.config import load_store_config
    from retail_vision.simulation.evaluate import evaluate_identities
    from retail_vision.simulation.world import build_demo_world
    from retail_vision.sinks import MemorySink

    cfg = load_store_config(config)
    world = build_demo_world(seed=seed)
    world.noise_std = noise
    runtime = _runtime_with_demo_gallery(
        cfg, world, detector, dropout=dropout, seed=seed, sink=MemorySink()
    )
    report = evaluate_identities(runtime, world, frames).as_dict()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
    _echo(report)


@app.command()
def render(
    config: Path = typer.Option("configs/store_demo.yaml"),
    frames: int = 300,
    out_dir: Path = typer.Option("data/frames", help="Where annotated frames / video go."),
    every: int = typer.Option(1, help="Write every Nth frame as PNG (0 = video only)."),
    video: bool = typer.Option(True, help="Also write an MP4 per camera."),
    detector: str = typer.Option("oracle", help="oracle (GT + noise) | blob (pixel-based mock)."),
    seed: int = 0,
) -> None:
    """Simulate and draw boxes, identities and zones on each camera (for demos/docs)."""
    from retail_vision.config import load_store_config
    from retail_vision.simulation.evaluate import evaluate_identities
    from retail_vision.simulation.render import render_simulation
    from retail_vision.simulation.world import build_demo_world
    from retail_vision.sinks import MemorySink

    cfg = load_store_config(config)
    # First pass: score identities against ground truth so the dashboard can show it.
    world = build_demo_world(seed=seed)
    runtime = _runtime_with_demo_gallery(cfg, world, detector, seed=seed, sink=MemorySink())
    report = evaluate_identities(runtime, world, frames).as_dict()
    # Second pass: same seed, render.
    world = build_demo_world(seed=seed)
    runtime = _runtime_with_demo_gallery(cfg, world, detector, seed=seed, sink=MemorySink())
    _echo(
        render_simulation(
            runtime, world, frames, out_dir, every=every, video=video, eval_report=report
        )
    )


@app.command("run-camera")
def run_camera(
    camera_id: str,
    config: Path = typer.Option(..., help="Store config with a video/RTSP source for the camera."),
    max_frames: int | None = None,
) -> None:
    """Run one camera worker over its configured file/RTSP source (real deployments)."""
    from retail_vision.config import load_store_config
    from retail_vision.pipeline.runner import StoreRuntime

    runtime = StoreRuntime(load_store_config(config))
    n = runtime.run_camera(camera_id, max_frames)
    runtime.close()
    _echo({"camera_id": camera_id, "frames": n, "metrics": runtime.metrics()[camera_id]})


@app.command("build-gallery")
def build_gallery(
    photos_root: Path = typer.Argument(..., help="<root>/<employee_id>/*.jpg"),
    config: Path = typer.Option("configs/store_demo.yaml"),
    output: Path | None = typer.Option(None, help="Defaults to reid.gallery_path in the config."),
) -> None:
    """Enrol employees: embed consented photos into the authorised gallery."""
    from retail_vision.config import load_store_config
    from retail_vision.reid.embedder import build_embedder
    from retail_vision.training.enrol import build_employee_gallery

    cfg = load_store_config(config)
    embedder = build_embedder(cfg.reid)
    gallery, stats = build_employee_gallery(
        photos_root, embedder, cfg.reid.max_embeddings_per_identity
    )
    out = output or Path(cfg.reid.gallery_path or "data/employee_gallery.npz")
    gallery.save(out)
    _echo({"gallery": str(out), "identities": len(gallery), "photos_per_identity": stats})


@app.command("reid-thresholds")
def reid_thresholds(
    gallery_path: Path = typer.Argument(..., help="Gallery .npz built with build-gallery."),
    target_far: float = 0.01,
    review_far: float = 0.05,
) -> None:
    """Pick match/review thresholds from the gallery impostor distribution; report ReID metrics."""
    from retail_vision.reid.gallery import Gallery
    from retail_vision.training.enrol import gallery_arrays
    from retail_vision.training.reid_eval import choose_thresholds, evaluate_reid

    embs, ids = gallery_arrays(Gallery.load(gallery_path))
    # leave-one-out style: each embedding queries the rest
    metrics = _loo_reid(embs, ids, evaluate_reid)
    _echo({"thresholds": choose_thresholds(embs, ids, target_far, review_far), "metrics": metrics})


@app.command()
def augment(
    images: Path,
    labels: Path,
    out: Path,
    factor: int = typer.Option(20, help="Augmented copies per source image."),
    image_size: int = 640,
    strength: str = typer.Option("medium", help="light | medium | heavy"),
    seed: int = 0,
) -> None:
    """Expand a small YOLO-format dataset with CCTV-oriented, bbox-aware augmentations."""
    from retail_vision.training.augment import expand_dataset

    _echo(expand_dataset(images, labels, out, factor, image_size, strength, seed))


@app.command("train-detector")
def train_detector_cmd(
    data_yaml: Path,
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    export: bool = typer.Option(True, help="Export best.pt to ONNX after training."),
) -> None:
    """Fine-tune a YOLO detector (requires the `training` extra and a GPU for sane speed)."""
    from retail_vision.training.detector import evaluate_detector, export_detector, train_detector

    best = train_detector(data_yaml, base_model, epochs, imgsz, batch, device)
    result = {"weights": str(best), "metrics": evaluate_detector(best, data_yaml, imgsz)}
    if export:
        result["onnx"] = str(export_detector(best, "onnx", imgsz))
    _echo(result)


@app.command("serve-api")
def serve_api(
    host: str = "0.0.0.0",
    port: int = 8000,
    db_path: str = typer.Option(":memory:", help="DuckDB file; :memory: for ephemeral."),
) -> None:
    """Start the cloud ingest/analytics API."""
    import uvicorn

    from retail_vision.cloud.api import create_app

    uvicorn.run(create_app(db_path), host=host, port=port)


@app.command()
def report(
    events: Path = typer.Argument("data/events.jsonl", help="Events JSONL from the edge workers."),
    output: Path | None = typer.Option(None, help="Write the report JSON here."),
) -> None:
    """Load events into DuckDB and print the store analytics report."""
    from retail_vision.cloud.analytics import EventStore

    store = EventStore()
    n = store.load_jsonl(events)
    out = {
        "loaded_events": n,
        "summary": store.summary(),
        "visits": store.store_visits(),
        "zone_dwell": store.zone_dwell(),
        "shelf_engagement": store.shelf_engagement(),
        "employee_presence": store.employee_presence(),
        "cross_camera_journeys": store.cross_camera_journeys(),
        "review_queue_size": len(store.review_queue(1000)),
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(out, indent=2, default=str))
    _echo(out)


# Helpers ---------------------------------------------------------------------------------


def _runtime_with_demo_gallery(
    cfg, world, detector: str = "oracle", dropout: float = 0.0, seed: int = 0, sink=None
):
    """Build a StoreRuntime whose employee gallery is enrolled from the synthetic actors and
    whose detectors see the synthetic world (oracle with noise, or the pixel-based blob mock)."""
    from retail_vision.detection.mock import ColorBlobDetector
    from retail_vision.detection.oracle import OracleDetector
    from retail_vision.pipeline.runner import StoreRuntime
    from retail_vision.reid.embedder import build_embedder
    from retail_vision.reid.gallery import Gallery

    embedder = build_embedder(cfg.reid)
    gallery = Gallery(embedder.dim, cfg.reid.max_embeddings_per_identity)
    for actor in world.actors:
        if actor.is_employee:
            gallery.add(actor.name, embedder.embed(world.render_enrolment_crop(actor)))
    runtime = StoreRuntime(cfg, sink=sink, employee_gallery=gallery)
    if cfg.detector.backend == "mock":
        for i, (camera_id, worker) in enumerate(runtime.workers.items()):
            if detector == "oracle":
                worker.detector = OracleDetector(
                    world, camera_id, dropout_rate=dropout, seed=seed + i
                )
            elif detector == "blob":
                worker.detector = ColorBlobDetector(
                    dropout_rate=dropout,
                    seed=seed + i,
                    background=world.render_background(camera_id),
                )
            else:
                raise typer.BadParameter("detector must be 'oracle' or 'blob'")
    return runtime


def _loo_reid(embs, ids, evaluate_reid):
    import numpy as np

    sims = embs @ embs.T
    np.fill_diagonal(sims, -1.0)
    order = np.argsort(-sims, axis=1)
    ranked = ids[order]
    hits = ranked == ids[:, None]
    return {
        "rank1": round(float(hits[:, 0].mean()), 4),
        "rank5": round(float(hits[:, :5].any(axis=1).mean()), 4),
        "samples": int(len(ids)),
        "identities": int(len(set(ids.tolist()))),
    }


if __name__ == "__main__":
    app()
