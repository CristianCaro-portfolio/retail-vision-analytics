# ADR 0001: Edge-first inference, events-only to the cloud

Status: accepted

## Context

6,000 cameras (300 stores x 20) recording continuously. Cloud inference would mean
streaming every feed, storing video, and paying for GPU minutes proportional to wall-clock
time regardless of how many people are on the floor.

## Decision

Run detection, tracking, re-identification and event logic on a box inside each store.
Ship only structured events (and optionally low-confidence crops for labelling) to the
cloud. The cloud owns analytics, monitoring, model training and gallery management.

## Consequences

- Network and storage cost drops by orders of magnitude; privacy posture improves because
  video never leaves the premises.
- Edge models must be small and exportable (ONNX / TensorRT); the detector interface is
  backend-agnostic for this reason.
- Identity resolution is per store by construction. Cross-store identity is not a goal.
- Fleet management (model rollout, health) becomes the operational core: hence per-camera
  metrics and heartbeats as first-class outputs.
