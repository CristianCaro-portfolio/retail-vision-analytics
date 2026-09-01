"""Draw the pipeline's view of the world: zones, boxes, identities, live counters and a
store-level dashboard that combines every camera with the cloud analytics."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from retail_vision.cloud.analytics import EventStore
from retail_vision.pipeline.runner import StoreRuntime
from retail_vision.simulation.sources import SyntheticSource
from retail_vision.simulation.world import StoreWorld
from retail_vision.types import Identity

FONT = cv2.FONT_HERSHEY_SIMPLEX
ZONE_COLORS = {
    "entrance": (90, 200, 90),
    "shelf": (40, 170, 255),
    "seat": (220, 120, 255),
    "area": (170, 170, 170),
}
EMPLOYEE_COLOR = (255, 150, 30)
CUSTOMER_COLOR = (60, 220, 255)
REVIEW_COLOR = (60, 60, 255)
PANEL = (28, 26, 24)
TILE = (42, 40, 38)
TEXT = (235, 235, 235)
MUTED = (150, 150, 150)


def _panel(img, x1, y1, x2, y2, alpha=0.65, color=PANEL):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _text(img, txt, org, scale=0.46, color=TEXT, thick=1):
    cv2.putText(img, txt, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_frame(frame: np.ndarray, worker, events_so_far: list) -> np.ndarray:
    out = frame.copy()
    overlay = out.copy()
    for zone in worker.events.zones:
        color = ZONE_COLORS.get(zone.kind, ZONE_COLORS["area"])
        pts = zone.polygon.astype(np.int32)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(out, [pts], True, color, 2)
        x, y = pts[:, 0, 0].min(), pts[:, 0, 1].min()
        _text(out, zone.name, (int(x) + 5, int(y) + 17), 0.5, color, 2)
    cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)

    for bbox, identity in worker.last_assignments:
        _draw_person(out, bbox, identity)

    h, w = out.shape[:2]
    m = worker.metrics.snapshot()
    _panel(out, 0, h - 44, w, h)
    _text(out, worker.camera.camera_id.upper(), (10, h - 26), 0.55, TEXT, 2)
    _text(
        out,
        f"frame {worker.frame_idx}   {m['fps']} fps   p95 {m['latency_ms_p95']} ms",
        (10, h - 9),
        0.42,
        MUTED,
    )
    right = (
        f"people {len(worker.last_assignments)}   events {m['events']}   "
        f"review {m['review_ratio']:.0%}"
    )
    (tw, _), _ = cv2.getTextSize(right, FONT, 0.45, 1)
    _text(out, right, (w - tw - 10, h - 17), 0.45)

    recent = events_so_far[-4:]
    if recent:
        _panel(out, w - 330, 0, w, 14 + 17 * len(recent), alpha=0.6)
        for i, e in enumerate(recent):
            who = e.employee_id or e.global_id
            color = EMPLOYEE_COLOR if e.role == "employee" else CUSTOMER_COLOR
            _text(
                out, f"{e.event_type}  {e.zone or ''}  {who}", (w - 322, 14 + i * 17), 0.42, color
            )
    return out


def _draw_person(img: np.ndarray, bbox, identity: Identity) -> None:
    x1, y1, x2, y2 = (int(v) for v in bbox.as_array())
    color = EMPLOYEE_COLOR if identity.role.value == "employee" else CUSTOMER_COLOR
    if identity.needs_review:
        color = REVIEW_COLOR
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = identity.employee_id or identity.global_id
    if identity.needs_review:
        label += " ?"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.48, 1)
    cv2.rectangle(img, (x1 - 1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    _text(img, label, (x1 + 3, y1 - 4), 0.48, (20, 20, 20))
    fx, fy = (int(v) for v in bbox.foot_point)
    cv2.circle(img, (fx, fy), 3, color, -1)


def draw_dashboard(
    cam_frames: dict[str, np.ndarray],
    store: EventStore,
    runtime: StoreRuntime,
    timestamp: float,
    eval_report: dict | None = None,
) -> np.ndarray:
    """Cameras on top, store KPIs + cross-camera journeys + shelf conversion below."""
    top = np.hstack([cam_frames[c.camera_id] for c in runtime.store.cameras])
    w = top.shape[1]
    panel_h = 330
    board = np.full((panel_h, w, 3), PANEL, dtype=np.uint8)

    s = store.summary()
    active = runtime.resolver.active_identities()
    on_floor = sum(1 for a in active.values() if timestamp - a["last_seen"] < 3)
    shelves = store.shelf_engagement()
    journeys = store.cross_camera_journeys(limit=6)
    visits = store.store_visits()
    review_ratio = float(np.mean([m["review_ratio"] for m in runtime.metrics().values()]))
    rate = float(np.mean([r["interaction_rate_pct"] for r in shelves])) if shelves else 0.0

    kpis = [
        ("VISITORS", str(visits[0]["visitors"] if visits else 0)),
        ("ON FLOOR NOW", str(on_floor)),
        ("EMPLOYEES ID'D", str(s["employees_seen"])),
        ("SHELF INTERACTION", f"{rate:.0f}%"),
        ("EVENTS", str(s["events"])),
        ("NEEDS REVIEW", f"{review_ratio:.1%}"),
    ]
    tile_w = w // len(kpis)
    for i, (label, value) in enumerate(kpis):
        x = i * tile_w
        cv2.rectangle(board, (x + 10, 12), (x + tile_w - 10, 92), TILE, -1)
        _text(board, label, (x + 22, 36), 0.42, MUTED)
        _text(board, value, (x + 22, 76), 1.0, TEXT, 2)

    x0, y0 = 20, 122
    _text(board, "CROSS-CAMERA JOURNEYS  (one id per person)", (x0, y0), 0.5)
    for i, j in enumerate(journeys):
        color = EMPLOYEE_COLOR if j["role"] == "employee" else CUSTOMER_COLOR
        y = y0 + 24 + i * 22
        cv2.circle(board, (x0 + 6, y - 5), 5, color, -1)
        _text(
            board,
            f"{j['global_id']:<14} {j['camera_list']:<15} {j['journey_s']:>6.1f} s",
            (x0 + 20, y),
        )

    x1 = w // 2 - 70
    _text(board, "SHELF ENGAGEMENT  (visited -> interacted)", (x1, y0), 0.5)
    for i, r in enumerate(shelves):
        y = y0 + 30 + i * 46
        _text(board, r["zone"], (x1, y))
        bar_w = 170
        cv2.rectangle(board, (x1 + 90, y - 12), (x1 + 90 + bar_w, y + 2), (60, 58, 56), -1)
        filled = int(bar_w * r["interaction_rate_pct"] / 100)
        cv2.rectangle(board, (x1 + 90, y - 12), (x1 + 90 + filled, y + 2), ZONE_COLORS["shelf"], -1)
        _text(
            board,
            f"{r['interacting']}/{r['visitors']}  {r['interaction_rate_pct']:.0f}%",
            (x1 + 100 + bar_w, y),
        )

    x2 = w - 310
    _text(board, "EMPLOYEES ON SHIFT", (x2, y0), 0.5)
    for i, e in enumerate(store.employee_presence()):
        y = y0 + 24 + i * 22
        cv2.circle(board, (x2 + 6, y - 5), 5, EMPLOYEE_COLOR, -1)
        seated = f"seated {e['seated_s']:.0f}s" if e["seated_s"] else ""
        row = f"{e['employee_id']}  {e['cameras']} cams  {e['zones']} zones  {seated}"
        _text(board, row, (x2 + 20, y), 0.42)
    if eval_report:
        y = y0 + 92
        _text(board, "IDENTITY QUALITY vs GROUND TRUTH", (x2, y), 0.5)
        lines = [
            f"role accuracy         {eval_report['role_accuracy']:.1%}",
            f"employee id accuracy  {eval_report['employee_id_accuracy']:.1%}",
            f"identity purity       {eval_report['purity']:.1%}",
            f"ids per person        {eval_report['mean_fragmentation']:.2f}",
        ]
        for i, line in enumerate(lines):
            _text(board, line, (x2, y + 24 + i * 20))

    _text(
        board,
        f"store {runtime.store.store_id}   t = {timestamp:5.1f} s   edge -> events only -> cloud",
        (20, panel_h - 12),
        0.42,
        MUTED,
    )
    return np.vstack([top, board])


def render_simulation(
    runtime: StoreRuntime,
    world: StoreWorld,
    frames: int,
    out_dir: str | Path,
    every: int = 1,
    video: bool = True,
    dashboard_every: int = 10,
    eval_report: dict | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        cam.camera_id: SyntheticSource(world, cam.camera_id, cam.fps)
        for cam in runtime.store.cameras
    }
    store = EventStore()
    fps = runtime.store.cameras[0].fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers: dict[str, cv2.VideoWriter] = {}
    if video:
        for cam in runtime.store.cameras:
            writers[cam.camera_id] = cv2.VideoWriter(
                str(out_dir / f"{cam.camera_id}.mp4"), fourcc, fps, (cam.width, cam.height)
            )
    dash_writer: cv2.VideoWriter | None = None
    events_by_cam: dict[str, list] = {c: [] for c in sources}
    written = 0
    for f in range(frames):
        ts = 0.0
        drawn_frames: dict[str, np.ndarray] = {}
        for camera_id, src in sources.items():
            ts, frame = src.current()
            worker = runtime.workers[camera_id]
            new_events = worker.process_frame(ts, frame)
            events_by_cam[camera_id].extend(new_events)
            store.insert(e.to_dict() for e in new_events)
            drawn = draw_frame(frame, worker, events_by_cam[camera_id])
            drawn_frames[camera_id] = drawn
            if camera_id in writers:
                writers[camera_id].write(drawn)
            if every and f % every == 0:
                cv2.imwrite(str(out_dir / f"{camera_id}_{f:05d}.png"), drawn)
                written += 1
        dash = draw_dashboard(drawn_frames, store, runtime, ts, eval_report)
        if video:
            if dash_writer is None:
                dash_writer = cv2.VideoWriter(
                    str(out_dir / "dashboard.mp4"), fourcc, fps, (dash.shape[1], dash.shape[0])
                )
            dash_writer.write(dash)
        if dashboard_every and f % dashboard_every == 0:
            cv2.imwrite(str(out_dir / f"dashboard_{f:05d}.png"), dash)
        runtime.resolver.expire(ts)
        runtime.dedup.prune(ts)
        world.step()
    for wr in writers.values():
        wr.release()
    if dash_writer is not None:
        dash_writer.release()
    return {
        "frames": frames,
        "pngs": written,
        "videos": sorted(str(p) for p in out_dir.glob("*.mp4")),
    }
