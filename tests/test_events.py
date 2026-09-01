from retail_vision.config import CameraConfig, EventConfig, ZoneConfig
from retail_vision.events.dedup import EventDeduplicator
from retail_vision.events.zones import ZoneEventEngine
from retail_vision.types import Event, Identity, PersonRole


def make_engine(**overrides):
    cam = CameraConfig(
        camera_id="cam-1",
        zones=[
            ZoneConfig(
                name="shelf", kind="shelf", polygon=[[0, 0], [100, 0], [100, 100], [0, 100]]
            ),
            ZoneConfig(
                name="desk", kind="seat", polygon=[[200, 0], [300, 0], [300, 100], [200, 100]]
            ),
        ],
    )
    cfg = EventConfig(
        zone_enter_frames=2, zone_exit_frames=2, min_dwell_seconds=1.0, interaction_frames=4
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return ZoneEventEngine("store", cam, cfg)


def run(engine, identity, points, fps=10.0):
    events = []
    for i, p in enumerate(points):
        events.extend(engine.update(identity, p, i / fps))
    return events


def test_enter_interaction_exit_and_dwell_sequence():
    engine = make_engine()
    ident = Identity("person-1", PersonRole.CUSTOMER)
    inside, outside = (50, 50), (150, 50)
    events = run(engine, ident, [inside] * 20 + [outside] * 5)
    types = [e.event_type for e in events]
    assert types == ["zone_entered", "product_interaction", "zone_exited", "dwell"]
    exit_event = events[2]
    assert exit_event.zone == "shelf"
    assert 1.7 <= exit_event.payload["dwell_seconds"] <= 2.1


def test_hysteresis_ignores_single_frame_flicker():
    engine = make_engine()
    ident = Identity("person-1", PersonRole.CUSTOMER)
    inside, outside = (50, 50), (150, 50)
    events = run(engine, ident, [inside, inside, inside, outside, inside, inside, inside])
    assert [e.event_type for e in events] == ["zone_entered"]


def test_seat_zone_emits_seat_events_with_employee_id():
    engine = make_engine()
    ident = Identity("emp-7", PersonRole.EMPLOYEE, employee_id="emp-7", confidence=0.95)
    events = run(engine, ident, [(250, 50)] * 15 + [(400, 50)] * 3)
    assert [e.event_type for e in events] == ["seat_occupied", "seat_released", "dwell"]
    assert all(e.employee_id == "emp-7" for e in events)
    assert events[0].payload["zone_kind"] == "seat"


def test_flush_missing_closes_open_visits():
    engine = make_engine()
    ident = Identity("person-1", PersonRole.CUSTOMER)
    run(engine, ident, [(50, 50)] * 15)
    closing = engine.flush_missing(timestamp=10.0, missing_after=1.0)
    assert [e.event_type for e in closing] == ["zone_exited", "dwell"]
    assert closing[0].global_id == "person-1"


def test_deduplicator_suppresses_repeats_within_window():
    dedup = EventDeduplicator(window_seconds=5)

    def ev(ts, cam="cam-1"):
        return Event("zone_entered", "s", cam, ts, "person-1", "customer", zone="shelf")

    assert dedup.accept(ev(0.0))
    assert not dedup.accept(ev(2.0, cam="cam-2"))  # other camera, same person/zone
    assert dedup.accept(ev(6.0))
