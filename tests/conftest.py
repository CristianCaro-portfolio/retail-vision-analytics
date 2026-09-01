from pathlib import Path

import pytest

from retail_vision.config import StoreConfig, load_store_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def store_config() -> StoreConfig:
    cfg = load_store_config(ROOT / "configs" / "store_demo.yaml")
    cfg.sink.kind = "memory"
    cfg.reid.gallery_path = None
    return cfg
