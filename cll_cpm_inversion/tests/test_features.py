"""Unit tests for feature extraction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cll_cpm_inversion.features import (
    OPERATIONAL_FEATURES,
    extract_features_from_mask,
    features_from_folder,
)


def _disk_mask(radius: int = 30, size: int = 100) -> np.ndarray:
    y, x = np.ogrid[:size, :size]
    cy = cx = size // 2
    return ((y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2).astype(np.uint8) * 255


def test_disk_features_match_geometry():
    """A clean disk has known features: circularity ~1, solidity ~1."""
    r = 30
    mask = _disk_mask(radius=r, size=200)
    feats = extract_features_from_mask(mask)
    assert set(feats) == set(OPERATIONAL_FEATURES)
    assert feats["circularity"] > 0.88
    assert feats["solidity"] > 0.97
    # Area approximately pi r^2
    assert abs(feats["total_area"] - np.pi * r ** 2) / (np.pi * r ** 2) < 0.05


def test_empty_mask_returns_nan():
    feats = extract_features_from_mask(np.zeros((50, 50), dtype=np.uint8))
    assert all(np.isnan(v) for v in feats.values())


def test_rgb_mask_is_handled():
    """3-channel images should be converted to grayscale and binarised."""
    mask = _disk_mask(radius=25, size=120)
    rgb = np.stack([mask, mask, mask], axis=-1)
    feats = extract_features_from_mask(rgb)
    assert feats["circularity"] > 0.85


def test_largest_component_wins():
    """Pinprick noise next to a big disk should be discarded."""
    big = _disk_mask(radius=30, size=200)
    big[5:7, 5:7] = 255
    big[10, 10] = 255
    feats = extract_features_from_mask(big)
    assert feats["solidity"] > 0.95


def test_features_from_folder_flat(tmp_path: Path):
    """Flat folder: one mask per file, one row per file."""
    for i, r in enumerate([20, 25, 30]):
        Image.fromarray(_disk_mask(radius=r, size=150)).save(
            tmp_path / f"spheroid_{i:02d}.png")
    df = features_from_folder(tmp_path)
    assert len(df) == 3
    assert {"spheroid_id"} <= set(df.columns)
    assert df["circularity"].min() > 0.85


def test_features_from_folder_nested(tmp_path: Path):
    """Nested: one subfolder per spheroid -> trajectory mean."""
    for sid in ["W001", "W002"]:
        sd = tmp_path / sid
        sd.mkdir()
        for frame, r in enumerate([22, 26, 30]):
            Image.fromarray(_disk_mask(radius=r, size=150)).save(
                sd / f"frame_{frame:02d}.png")
    df = features_from_folder(tmp_path)
    assert len(df) == 2
    assert set(df["spheroid_id"]) == {"W001", "W002"}
    assert "n_frames" in df.columns
    assert (df["n_frames"] == 3).all()


def test_no_aggregate_returns_one_row_per_frame(tmp_path: Path):
    sd = tmp_path / "W001"
    sd.mkdir()
    for frame, r in enumerate([22, 26, 30, 34]):
        Image.fromarray(_disk_mask(radius=r, size=150)).save(
            sd / f"frame_{frame:02d}.png")
    df = features_from_folder(tmp_path, aggregate=False)
    assert len(df) == 4
    assert "frame" in df.columns
