import numpy as np
import pytest

from retail_vision.config import ReIDConfig
from retail_vision.reid.embedder import HistogramEmbedder, l2_normalize
from retail_vision.reid.gallery import Gallery
from retail_vision.reid.resolver import IdentityResolver
from retail_vision.training.reid_eval import choose_thresholds, evaluate_reid
from retail_vision.types import PersonRole


def unit(v):
    return l2_normalize(np.array(v, dtype=np.float32))


@pytest.fixture
def cfg():
    return ReIDConfig(match_threshold=0.9, review_threshold=0.7, identity_ttl_seconds=5)


def test_histogram_embedder_separates_colours_and_is_unit_norm():
    emb = HistogramEmbedder()
    red = np.zeros((90, 36, 3), np.uint8)
    red[:] = (0, 0, 255)
    blue = np.zeros((90, 36, 3), np.uint8)
    blue[:] = (255, 0, 0)
    e_red, e_blue = emb.embed(red), emb.embed(blue)
    assert abs(np.linalg.norm(e_red) - 1) < 1e-5
    assert float(e_red @ emb.embed(red.copy())) > 0.99
    assert float(e_red @ e_blue) < 0.6


def test_gallery_roundtrip(tmp_path):
    g = Gallery(dim=3)
    g.add("emp-1", unit([1, 0, 0]), {"name": "one"})
    g.add("emp-2", unit([0, 1, 0]))
    g.save(tmp_path / "gal.npz")
    loaded = Gallery.load(tmp_path / "gal.npz")
    assert set(loaded.ids()) == {"emp-1", "emp-2"}
    assert loaded.meta("emp-1") == {"name": "one"}
    best_id, sim = loaded.query(unit([0.9, 0.1, 0]))[0]
    assert best_id == "emp-1" and sim > 0.9


def test_employee_match_review_and_unknown_bands(cfg):
    gallery = Gallery(dim=3)
    gallery.add("emp-1", unit([1, 0, 0]))
    resolver = IdentityResolver(cfg, 3, gallery)

    ident = resolver.resolve("cam-1", 1, PersonRole.EMPLOYEE, unit([1, 0.05, 0]), 0.0)
    assert ident.employee_id == "emp-1" and not ident.needs_review

    ident = resolver.resolve("cam-1", 2, PersonRole.EMPLOYEE, unit([0.8, 0.6, 0]), 0.0)
    assert ident.employee_id == "emp-1" and ident.needs_review  # in the review band

    ident = resolver.resolve("cam-1", 3, PersonRole.EMPLOYEE, unit([0, 0, 1]), 0.0)
    assert ident.employee_id is None and ident.needs_review
    assert ident.global_id.startswith("employee-unknown-")


def test_customer_keeps_one_id_across_cameras_and_expires(cfg):
    resolver = IdentityResolver(cfg, 3)
    a = resolver.resolve("cam-1", 1, PersonRole.CUSTOMER, unit([0, 1, 0]), 0.0)
    b = resolver.resolve("cam-2", 7, PersonRole.CUSTOMER, unit([0, 1, 0.02]), 1.0)
    assert a.global_id == b.global_id  # cross-camera ReID
    c = resolver.resolve("cam-2", 8, PersonRole.CUSTOMER, unit([1, 0, 0]), 1.0)
    assert c.global_id != a.global_id
    assert (
        resolver.expire(now=1.0 + cfg.identity_ttl_seconds + 1)
        == sorted([a.global_id, c.global_id])
        or set(resolver.expire(1.0 + cfg.identity_ttl_seconds + 1)) == set()
    )
    assert resolver.active_identities() == {}


def test_track_binding_is_sticky_and_released(cfg):
    gallery = Gallery(dim=3)
    gallery.add("emp-1", unit([1, 0, 0]))
    resolver = IdentityResolver(cfg, 3, gallery, reverify_every=100)
    first = resolver.resolve("cam-1", 1, PersonRole.EMPLOYEE, unit([1, 0, 0]), 0.0)
    # same track, garbage embedding (occlusion): binding must hold
    second = resolver.resolve("cam-1", 1, PersonRole.EMPLOYEE, unit([0, 0, 1]), 0.1)
    assert second.global_id == first.global_id == "emp-1"
    resolver.release_track("cam-1", 1)
    third = resolver.resolve("cam-1", 1, PersonRole.EMPLOYEE, unit([0, 0, 1]), 0.2)
    assert third.global_id != "emp-1"


def test_reid_metrics_and_threshold_selection():
    rng = np.random.default_rng(0)
    centers = l2_normalize(rng.normal(size=(5, 16)).astype(np.float32))
    embs, ids = [], []
    for i, c in enumerate(centers):
        for _ in range(6):
            embs.append(l2_normalize(c + 0.15 * rng.normal(size=16).astype(np.float32)))
            ids.append(f"id-{i}")
    embs, ids = np.stack(embs), np.array(ids)
    m = evaluate_reid(embs[::2], ids[::2], embs[1::2], ids[1::2])
    assert m.rank1 > 0.9 and m.map > 0.8
    t = choose_thresholds(embs, ids, target_far=0.01, review_far=0.05)
    assert t["review_threshold"] <= t["match_threshold"]
    assert t["genuine_accept_at_match"] > 0.8
