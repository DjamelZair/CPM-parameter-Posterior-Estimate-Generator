"""Unit tests for the matcher + posterior summary."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cll_cpm_inversion.features import OPERATIONAL_FEATURES
from cll_cpm_inversion.invert import (
    IDENTIFIABILITY_THRESHOLDS,
    PARAMS,
    feature_weights_from_sobol,
    identifiability_flag,
    invert_observations,
    load_identifiability,
    load_sobol_indices,
    load_synthetic_library,
    summarise_posterior,
)


def test_library_loads():
    lib = load_synthetic_library()
    assert len(lib) >= 580
    assert set(PARAMS) <= set(lib.columns)
    assert set(OPERATIONAL_FEATURES) <= set(lib.columns)


def test_sobol_loads():
    sob = load_sobol_indices()
    assert {"feature", "parameter", "S1", "ST"} <= set(sob.columns)
    assert set(OPERATIONAL_FEATURES) <= set(sob["feature"])


def test_identifiability_loads():
    loo = load_identifiability()
    assert set(PARAMS) == set(loo["param"])
    assert (loo["R2"].between(-1.0, 1.0)).all()


def test_feature_weights_sum_to_one():
    w = feature_weights_from_sobol()
    assert set(w.keys()) == set(OPERATIONAL_FEATURES)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(v >= 0 for v in w.values())


def test_identifiability_flag_thresholds():
    assert identifiability_flag(0.9) == "identifiable"
    assert identifiability_flag(0.5) == "weakly identifiable"
    assert identifiability_flag(0.1) == "unidentifiable"
    assert identifiability_flag(-0.2) == "unidentifiable"
    # Exact boundaries
    assert identifiability_flag(IDENTIFIABILITY_THRESHOLDS["identifiable"]) == "identifiable"
    assert identifiability_flag(IDENTIFIABILITY_THRESHOLDS["weak"]) == "weakly identifiable"


def test_invert_recovers_library_sample():
    """Feeding a synthetic vector back in should return that sample at rank 0."""
    lib = load_synthetic_library()
    pick = lib.iloc[42]
    feats = pd.DataFrame([{
        "spheroid_id": "self_42",
        **{f: pick[f] for f in OPERATIONAL_FEATURES},
    }])
    post = invert_observations(feats, k=20)
    rank0 = post[post["rank"] == 0].iloc[0]
    assert rank0["sample_id"] == lib.index[42]
    for p in PARAMS:
        assert rank0[p] == pytest.approx(pick[p])


def test_summarise_posterior_columns():
    lib = load_synthetic_library()
    feats = pd.DataFrame([{
        "spheroid_id": f"obs_{i}",
        **{f: lib.iloc[i][f] for f in OPERATIONAL_FEATURES},
    } for i in range(3)])
    post = invert_observations(feats, k=20)
    summary = summarise_posterior(post)
    assert len(summary) == 3 * len(PARAMS)
    expected_cols = {"spheroid_id", "parameter", "median", "q05", "q95",
                     "q25", "q75", "n_matches", "loo_r2", "identifiability"}
    assert expected_cols <= set(summary.columns)


def test_summary_identifiability_matches_loo():
    """The flag set on the summary must match what the LOO table says."""
    lib = load_synthetic_library()
    feats = pd.DataFrame([{
        "spheroid_id": "x",
        **{f: lib.iloc[0][f] for f in OPERATIONAL_FEATURES},
    }])
    post = invert_observations(feats, k=20)
    summary = summarise_posterior(post)
    loo = load_identifiability()
    for _, row in summary.iterrows():
        r2 = float(loo.loc[loo["param"] == row["parameter"], "R2"].iloc[0])
        assert row["identifiability"] == identifiability_flag(r2)


def test_missing_feature_column_raises():
    bad = pd.DataFrame([{"spheroid_id": "x", "total_area": 1000.0}])
    with pytest.raises(ValueError):
        invert_observations(bad, k=5)
