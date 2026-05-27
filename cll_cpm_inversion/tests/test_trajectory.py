"""Tests for the tau-registered trajectory matcher (primary method)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cll_cpm_inversion import (
    OPERATIONAL_FEATURES,
    infer_from_trajectory,
    infer_from_mask_trajectories,
    load_trajectory_library,
)


def test_library_shape():
    lib = load_trajectory_library()
    N, T, F = lib["reps"].shape
    assert N > 1000 and T == 50 and F == len(OPERATIONAL_FEATURES)
    assert lib["params"].shape == (N, 7)


def _toy_traj(sid, scale, n=10):
    # a monotone trajectory; values arbitrary but >0
    rows = []
    for t in range(n):
        rows.append({"spheroid_id": sid, "frame": t,
                     "total_area": 1e5 * scale * (1 - 0.3 * t / n),
                     "equivalent_diameter": 400 * scale,
                     "solidity": 0.9, "perimeter": 2000 * scale,
                     "circularity": 0.5 - 0.2 * t / n})
    return pd.DataFrame(rows)


def test_infer_from_trajectory_runs():
    df = pd.concat([_toy_traj("A", 1.0), _toy_traj("B", 1.2)], ignore_index=True)
    post = infer_from_trajectory(df, id_col="spheroid_id", frame_col="frame", k=20)
    assert post["spheroid_id"].nunique() == 2
    assert {"spheroid_id", "parameter", "median", "identifiability"} <= set(post.columns)
    assert post[post.spheroid_id == "A"]["parameter"].nunique() == 7


def test_single_frame_rejected():
    one = _toy_traj("A", 1.0, n=1)
    with pytest.raises(ValueError):
        infer_from_trajectory(one, id_col="spheroid_id", k=20)


def test_mask_trajectories_bundled_example():
    post = infer_from_mask_trajectories("examples/real_masks/", k=20)
    assert post["spheroid_id"].nunique() >= 4
    # identifiable adhesion parameters present with finite medians
    sub = post[(post.parameter == "contact") & post["median"].notna()]
    assert len(sub) >= 4
