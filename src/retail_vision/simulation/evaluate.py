"""Identity-quality evaluation against simulator ground truth.

Reports the metrics that matter for double counting and employee attribution:

- fragmentation: how many global ids each real person received (ideal 1)
- purity:        share of frames where a global id maps to its majority person
- role accuracy: employee/customer classification accuracy per frame
- employee id accuracy: for employee frames, share correctly attributed to the
                        right employee_id (and share flagged unknown/needs review)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from retail_vision.pipeline.runner import StoreRuntime
from retail_vision.simulation.sources import SyntheticSource
from retail_vision.simulation.world import StoreWorld
from retail_vision.types import BBox


@dataclass
class IdentityReport:
    frames: int
    matched_frames: int
    fragmentation: dict[str, int]
    purity: float
    role_accuracy: float
    employee_id_accuracy: float
    employee_unknown_ratio: float
    ids_per_person: dict[str, list[str]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "frames": self.frames,
            "matched_person_frames": self.matched_frames,
            "mean_fragmentation": round(
                sum(self.fragmentation.values()) / max(len(self.fragmentation), 1), 3
            ),
            "fragmentation": self.fragmentation,
            "purity": round(self.purity, 3),
            "role_accuracy": round(self.role_accuracy, 3),
            "employee_id_accuracy": round(self.employee_id_accuracy, 3),
            "employee_unknown_ratio": round(self.employee_unknown_ratio, 3),
            "ids_per_person": self.ids_per_person,
        }


def evaluate_identities(runtime: StoreRuntime, world: StoreWorld, frames: int) -> IdentityReport:
    sources = {
        cam.camera_id: SyntheticSource(world, cam.camera_id, cam.fps)
        for cam in runtime.store.cameras
    }
    pairs: list[tuple[str, str, bool, bool, str | None]] = []  # gt, gid, gt_emp, pred_emp, emp_id
    for _ in range(frames):
        ts = 0.0
        for camera_id, src in sources.items():
            ts, frame = src.current()
            worker = runtime.workers[camera_id]
            worker.process_frame(ts, frame)
            gt = world.ground_truth(camera_id)
            for bbox, identity in worker.last_assignments:
                best, best_iou = None, 0.3
                for g in gt:
                    iou = bbox.iou(BBox(*g["bbox"]))
                    if iou > best_iou:
                        best, best_iou = g, iou
                if best is None:
                    continue
                pairs.append(
                    (
                        best["name"],
                        identity.global_id,
                        best["is_employee"],
                        identity.role.value == "employee",
                        identity.employee_id,
                    )
                )
        runtime.resolver.expire(ts)
        world.step()

    ids_per_person: dict[str, Counter] = defaultdict(Counter)
    persons_per_id: dict[str, Counter] = defaultdict(Counter)
    role_ok = emp_frames = emp_ok = emp_unknown = 0
    for gt_name, gid, gt_emp, pred_emp, emp_id in pairs:
        ids_per_person[gt_name][gid] += 1
        persons_per_id[gid][gt_name] += 1
        role_ok += int(gt_emp == pred_emp)
        if gt_emp:
            emp_frames += 1
            if emp_id is None:
                emp_unknown += 1
            elif emp_id == gt_name:
                emp_ok += 1

    n = max(len(pairs), 1)
    purity = sum(c.most_common(1)[0][1] for c in persons_per_id.values()) / n
    # Ignore ids that appear in <2% of a person's frames: transient flickers, not fragments.
    fragmentation = {
        name: sum(1 for _, c in cnt.items() if c >= 0.02 * sum(cnt.values()))
        for name, cnt in ids_per_person.items()
    }
    return IdentityReport(
        frames=frames,
        matched_frames=len(pairs),
        fragmentation=fragmentation,
        purity=purity,
        role_accuracy=role_ok / n,
        employee_id_accuracy=emp_ok / max(emp_frames, 1),
        employee_unknown_ratio=emp_unknown / max(emp_frames, 1),
        ids_per_person={k: [g for g, _ in v.most_common()] for k, v in ids_per_person.items()},
    )
