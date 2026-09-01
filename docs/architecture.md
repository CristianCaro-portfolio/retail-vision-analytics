# Architecture

## The problem

A retail franchise runs about 20 cameras per store across hundreds of stores. The business
wants, per store and per employee: who entered, who sat where and for how long, which
shelves customers stop at and touch, and how staff move through the floor. Two things make
this hard at scale:

1. **Volume.** 300 stores x 20 cameras = 6,000 video streams. Streaming raw video to the
   cloud for inference is not viable on cost (network, storage, GPU) and often not on
   privacy either.
2. **Identity.** Twenty overlapping cameras see the same person many times. Without
   cross-camera re-identification every metric is double counted, and "is this Jones?"
   needs an identity layer on top of detection.

## Design principles

- **Edge first.** Video never leaves the store. Each store box runs one worker per camera:
  detect -> track -> classify role -> embed -> resolve identity -> zone events. Only small
  JSON events (and optionally low-confidence crops for labelling) go to the cloud.
- **Two-level identity.** Level 1 is the binary employee/customer decision (cheap, high
  recall: uniform, badge, or a fine-tuned class). Level 2 answers *who*: employees are
  matched against an authorised enrolment gallery built from consented photos; customers
  are matched only against an anonymous, short-lived online gallery so one shopper keeps
  one anonymous id across cameras and then expires.
- **Never force an identity.** Every decision carries a similarity score. Above
  `match_threshold` it is accepted; between `review_threshold` and `match_threshold` it is
  emitted as *needs review*; below it stays unknown. The review band is the retraining loop:
  those crops are the hard negatives that improve the ReID model.
- **Hysteresis everywhere.** Zone enter/exit use separate frame counts, identities have a
  TTL, events are de-duplicated per (identity, event, zone) in a time window. Flapping at
  zone borders and camera overlaps is the main source of wrong counts in practice.
- **Monitoring is the product.** Per-camera FPS, latency, people per frame, review ratio and
  an embedding-drift score are first-class outputs, because operating 6,000 cameras is
  mostly about knowing which ones went bad.

## Components

```
                       STORE (edge box)                                    CLOUD
+-----------------------------------------------------------------+   +-------------------------+
| cam-01 ---> CameraWorker ----+                                  |   |                         |
|   frame -> Detector          |                                  |   |  POST /v1/events        |
|         -> SortTracker       |      shared per store            |   |      |                  |
|         -> RoleClassifier    +--> IdentityResolver              |   |      v                  |
|         -> Embedder          |      employee gallery (enrolled) |   |  EventStore (DuckDB /   |
|         -> ZoneEventEngine   |      online gallery (TTL)        |   |   BigQuery / Snowflake) |
|         -> CameraMetrics     +--> EventDeduplicator             |   |      |                  |
| cam-02 ---> CameraWorker ----+        |                         |   |      v                  |
|   ...                                 v                         |   |  /v1/analytics/*        |
| cam-20 ---> CameraWorker ----> EventSink (jsonl | http batch) --+-->|  review queue           |
+-----------------------------------------------------------------+   |  camera health          |
                                                                      +-------------------------+
```

| Stage | Module | POC implementation | Production swap |
|---|---|---|---|
| Detection | `detection/` | `OracleDetector` (GT + noise), `ColorBlobDetector` (pixels) | `YoloDetector` (Ultralytics) or `OnnxDetector` (ONNX Runtime / TensorRT) |
| Tracking | `tracking/` | SORT: Kalman + Hungarian IoU (numpy/scipy only) | ByteTrack via `supervision`, same interface |
| Role | `reid/role.py` | uniform colour on the torso band | small classifier head or extra detector class |
| Embedding | `reid/embedder.py` | HSV histogram, 2 body parts | OSNet / FastReID exported to ONNX |
| Identity | `reid/resolver.py` | in-memory per store | same logic on Redis for multi-box stores |
| Events | `events/` | zone polygons + hysteresis + dedup | unchanged |
| Transport | `sinks/` | JSONL, HTTP batch with local buffering | HTTP -> Pub/Sub / Kinesis |
| Analytics | `cloud/` | FastAPI + DuckDB | same SQL on BigQuery / Snowflake |
| Monitoring | `monitoring/` | rolling metrics, drift score, Prometheus gauges | Prometheus + Grafana, alerts per camera |

## Identity resolution in detail

```
track (camera, track_id) -----------------------------------------------+
        |                                                               |
   bound already? --yes--> return bound identity (re-verify every N)    |
        | no                                                            |
   role == employee?                                                    |
     yes: query employee gallery (top-2, margin check)                  |
          sim >= match   -> employee_id, confident                      |
          sim >= review  -> employee_id, needs_review                   |
          else           -> employee-unknown-XXXX, needs_review         |
     no : query online gallery of active customers                      |
          sim >= match   -> reuse person-XXXX (cross-camera ReID)       |
          else           -> new person-XXXX                             |
        |                                                               |
   bind (camera, track_id) -> global_id; add embedding to online gallery |
   (only if it already resembles that identity: no gallery poisoning)   |
```

Crops that touch the frame border, are too small or too wide are never embedded (quality
gate). A person entering the frame is tracked immediately but only gets an identity once
seen whole; this single rule removed most of the fragmentation in the simulator.

## Training loop (offline)

1. Label ~100 frames per camera angle (person, employee) in YOLO format.
2. `rva augment` expands x20-x100 with CCTV-oriented, bbox-aware augmentation.
3. `rva train-detector` fine-tunes from a COCO checkpoint (backbone frozen for small data),
   evaluates mAP and exports ONNX for the edge.
4. `rva build-gallery` enrols employees from consented photos.
5. `rva reid-thresholds` picks `match_threshold` / `review_threshold` from the impostor
   similarity distribution at a target false-accept rate instead of guessing.
6. In production the review queue feeds new labelled crops back into 1 and 4.

## Scaling notes

- One edge box (a GPU-less mini PC or a Jetson) handles a store: 20 cameras at 5-10 fps
  with a nano detector on ONNX and embeddings computed every 3rd frame per track.
- The cloud only ingests events: ~1-5 events per person-minute per camera, a few KB.
  6,000 cameras is a modest stream for any managed queue + warehouse.
- Per-camera metrics roll up to a health board; drift score and review ratio are the two
  numbers that tell you a camera moved or the lighting changed before customers do.

## Privacy and fairness

- Raw video and faces stay in the store. Employee identification uses appearance
  embeddings against a gallery the employer built with consent; there is no face
  recognition in this repository.
- Customers are never identified, only re-identified anonymously for the duration of a
  visit (TTL), then forgotten.
- Thresholds must be evaluated per camera and, where legally appropriate, per subgroup;
  the review band exists so the system defaults to "unknown" instead of a wrong name.
