import cv2
import numpy as np

from retail_vision.training.augment import expand_dataset


def test_expand_dataset_writes_images_and_valid_labels(tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for i in range(2):
        img = np.full((480, 640, 3), 200, np.uint8)
        cv2.rectangle(img, (100, 100), (160, 260), (30, 30, 200), -1)
        cv2.imwrite(str(images / f"f{i}.jpg"), img)
        (labels / f"f{i}.txt").write_text("0 0.203125 0.375 0.09375 0.333333\n")

    stats = expand_dataset(images, labels, tmp_path / "out", factor=5, image_size=320, seed=1)
    assert stats["source_images"] == 2
    assert stats["written_images"] == 2 * (1 + 5)
    out_labels = sorted((tmp_path / "out" / "labels").glob("*.txt"))
    assert len(out_labels) == 12
    for lbl in out_labels:
        for line in lbl.read_text().splitlines():
            cls, *box = line.split()
            assert cls == "0"
            assert all(0.0 <= float(v) <= 1.0 for v in box)
    sample = cv2.imread(str(next((tmp_path / "out" / "images").glob("*_aug000.jpg"))))
    assert sample.shape[:2] == (320, 320)
