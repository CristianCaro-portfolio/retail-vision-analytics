import numpy as np

from retail_vision.config import TrackerConfig
from retail_vision.tracking.sort import SortTracker, iou_matrix
from retail_vision.types import BBox, Detection


def det(x1, y1, x2, y2, conf=0.9):
    return Detection(BBox(x1, y1, x2, y2), conf)


def test_iou_matrix_values():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], dtype=float)
    iou = iou_matrix(a, b)[0]
    assert iou[0] == 1.0
    assert abs(iou[1] - 25 / 175) < 1e-6
    assert iou[2] == 0.0


def test_track_id_is_stable_while_moving():
    tracker = SortTracker(TrackerConfig(min_hits=1))
    ids = set()
    for step in range(20):
        x = 10 + step * 5
        tracks = tracker.update([det(x, 10, x + 30, 90)])
        assert len(tracks) == 1
        ids.add(tracks[0].track_id)
    assert ids == {1}


def test_two_people_keep_distinct_ids_when_crossing():
    tracker = SortTracker(TrackerConfig(min_hits=1))
    history = {}
    for step in range(30):
        a = det(10 + step * 6, 10, 40 + step * 6, 90)  # moves right
        b = det(200 - step * 6, 12, 230 - step * 6, 92)  # moves left
        for t in tracker.update([a, b]):
            history.setdefault(t.track_id, []).append(t.bbox.x1)
    assert len(history) == 2
    xs = sorted(history.values(), key=lambda h: h[0])
    assert xs[0][-1] > xs[0][0]  # the track that started left ended right
    assert xs[1][-1] < xs[1][0]


def test_track_survives_short_occlusion_and_dies_after_max_age():
    tracker = SortTracker(TrackerConfig(min_hits=1, max_age=3))
    tracker.update([det(10, 10, 40, 90)])
    for _ in range(3):
        assert tracker.update([]) == []
    revived = tracker.update([det(12, 10, 42, 90)])
    assert revived and revived[0].track_id == 1
    for _ in range(4):
        tracker.update([])
    new = tracker.update([det(12, 10, 42, 90)])
    assert new[0].track_id == 2


def test_reported_box_is_the_raw_detection():
    tracker = SortTracker(TrackerConfig(min_hits=1))
    tracker.update([det(10, 10, 40, 90)])
    tracks = tracker.update([det(15, 10, 45, 92)])
    assert tracks[0].bbox.as_array().tolist() == [15, 10, 45, 92]
