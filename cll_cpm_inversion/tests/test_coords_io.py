"""Tests for the (cell_id, x, y) coordinate-CSV input path."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from cll_cpm_inversion.coords_io import (
    features_from_coords,
    mask_from_points,
    read_coords,
)
from cll_cpm_inversion.features import extract_features_from_mask
from cll_cpm_inversion import infer_from_coords


def _disk_coords(cx=40, cy=40, r=18, sid="S001") -> pd.DataFrame:
    yy, xx = np.mgrid[0:2 * (cy + r), 0:2 * (cx + r)]
    m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    ys, xs = np.where(m)
    return pd.DataFrame({"cell_id": sid, "x": xs, "y": ys})


def test_read_coords_aliases():
    df = pd.DataFrame({"label": ["A"], "col": [3], "row": [4]})
    out = read_coords(df)
    assert list(out.columns) == ["spheroid_id", "x", "y"]
    assert out.iloc[0].tolist() == ["A", 3, 4]


def test_read_coords_missing_column():
    with pytest.raises(KeyError):
        read_coords(pd.DataFrame({"id": ["A"], "x": [1]}))  # no y


def test_mask_translation_invariant():
    # the same disk shifted by a large offset rasterises to the same shape
    d = _disk_coords()
    m1 = mask_from_points(d.x, d.y)
    m2 = mask_from_points(d.x + 1000, d.y + 500)
    assert m1.sum() == m2.sum()


def test_disk_is_round():
    feats = features_from_coords(_disk_coords()).iloc[0]
    # a filled disk has circularity and solidity close to 1
    assert feats["circularity"] > 0.9
    assert feats["solidity"] > 0.95


def test_coords_match_mask_path():
    # features built from coords must equal features from the rasterised mask
    d = _disk_coords()
    mask = mask_from_points(d.x, d.y)
    f_mask = extract_features_from_mask(mask)
    f_coord = features_from_coords(d).iloc[0]
    for key, val in f_mask.items():
        assert f_coord[key] == pytest.approx(val, rel=1e-9, abs=1e-9)


def test_multiple_spheroids_and_inversion():
    d = pd.concat([_disk_coords(40, 40, 18, "S001"),
                   _disk_coords(30, 30, 11, "S002")], ignore_index=True)
    feats = features_from_coords(d)
    assert set(feats["spheroid_id"]) == {"S001", "S002"}
    post = infer_from_coords(d, k=20)
    # 7 CPM parameters per spheroid
    assert post["spheroid_id"].nunique() == 2
    assert set(post.columns) >= {"spheroid_id", "parameter", "median",
                                 "identifiability"}


def test_semicolon_delimited(tmp_path):
    d = _disk_coords()
    p = tmp_path / "coords.csv"
    d.to_csv(p, sep=";", index=False)
    feats = features_from_coords(str(p))  # sep auto-detected
    assert len(feats) == 1
