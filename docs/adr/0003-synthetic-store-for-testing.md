# ADR 0003: A synthetic store with ground truth for tests and evaluation

Status: accepted

## Context

Tracking, cross-camera ReID and zone logic are where retail analytics goes wrong, and they
are hard to test on real footage: no ground truth, privacy constraints, slow to iterate.
Detector quality is a separate concern with its own metrics (mAP).

## Decision

Ship a small simulator: actors with scripted paths in a shared world, several overlapping
camera viewports, fixtures and zones. Two detectors run on it:

- `OracleDetector`: ground truth plus realistic noise (jitter, misses, occlusion drops).
  Isolates identity/event logic from detector quality.
- `ColorBlobDetector`: background subtraction on real pixels, deliberately weak under
  occlusion. A robustness check for the same logic.

`rva evaluate` scores fragmentation (ids per person), purity, role accuracy and employee-id
accuracy against ground truth; CI asserts on them.

## Consequences

- The whole pipeline runs on any CPU in seconds with no model downloads.
- Regressions in identity logic are caught by numbers, not by eyeballing videos.
- Real-footage evaluation is still required before deployment; the simulator is not a
  substitute for a labelled validation set from the client's cameras.
