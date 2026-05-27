"""Trajectory (tau-registered) matching - the primary matcher.

Instead of collapsing a spheroid to a single end-state feature vector, this
compares the whole time course against the bundled library *trajectories* on a
shared phase axis ``tau in [0, 1]``: each trajectory is z-scored with the library
statistics, resampled to ``T`` phase points, and compared point by point with a
Sobol-weighted mean absolute difference. This is order-preserving and needs no
calibration between simulation steps and real time. The static end-state matcher
in :mod:`invert` is retained for comparison (see the thesis discussion on
static-versus-dynamic matching).

Bundled data: ``data/synthetic_library_trajectories.csv.gz`` (1105 samples x T
phase points x the operational features). Identifiability flags come from the
tau-registration leave-one-out benchmark in ``data/identifiability_loo.csv``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import OPERATIONAL_FEATURES, features_from_folder
from .invert import (PARAMS, feature_weights_from_sobol, load_identifiability,
                     load_synthetic_library, summarise_posterior)

T = 50


def _data_path(name: str) -> Path:
    return Path(__file__).resolve().parent / "data" / name


def load_trajectory_library():
    """Bundled library trajectories -> z-scored reps (N, T, F), params, stats."""
    traj = pd.read_csv(_data_path("synthetic_library_trajectories.csv.gz"))
    lib = load_synthetic_library()                       # sample_id index + PARAMS
    feats = OPERATIONAL_FEATURES
    samples = sorted(traj["sample_id"].unique())
    reps = np.stack([traj[traj["sample_id"] == s].sort_values("tau")[feats].to_numpy()
                     for s in samples])                  # (N, T, F)
    params = np.stack([lib.loc[s, PARAMS].to_numpy(dtype=float) for s in samples])
    flat = reps.reshape(-1, len(feats))
    mu, sd = flat.mean(axis=0), flat.std(axis=0) + 1e-9
    return {"samples": np.array(samples), "reps": (reps - mu) / sd,
            "params": params, "mu": mu, "sd": sd}


def _tau_rep(frame_feats: np.ndarray, mu, sd) -> np.ndarray:
    """(n_frames, F) trajectory -> (T, F) z-scored, resampled onto tau in [0,1]."""
    z = (frame_feats - mu) / sd
    src = np.linspace(0.0, 1.0, len(z))
    tau = np.linspace(0.0, 1.0, T)
    return np.stack([np.interp(tau, src, z[:, f]) for f in range(z.shape[1])], axis=1)


def infer_from_trajectory(frames_df: pd.DataFrame,
                          id_col: str = "spheroid_id",
                          frame_col: str | None = None,
                          k: int = 20) -> pd.DataFrame:
    """Invert per-frame trajectories into CPM parameter posteriors.

    Parameters
    ----------
    frames_df : DataFrame with `id_col`, the OPERATIONAL_FEATURES, and one row
        per frame. If `frame_col` is given the frames are ordered by it,
        otherwise the existing row order is used.
    Returns the same posterior summary as `infer_from_masks` / `infer_from_features`.
    """
    missing = [f for f in OPERATIONAL_FEATURES if f not in frames_df.columns]
    if missing:
        raise ValueError(f"frames_df missing required columns: {missing}")
    lib = load_trajectory_library()
    wd = feature_weights_from_sobol(features=OPERATIONAL_FEATURES)
    w = np.array([wd[f] for f in OPERATIONAL_FEATURES])

    rows = []
    for sid, g in frames_df.groupby(id_col, sort=False):
        if frame_col is not None:
            g = g.sort_values(frame_col)
        ff = g[OPERATIONAL_FEATURES].to_numpy(dtype=float)
        if len(ff) < 2 or np.isnan(ff).any():
            continue
        q = _tau_rep(ff, lib["mu"], lib["sd"])
        d = np.zeros(lib["reps"].shape[0])
        for fi in range(len(OPERATIONAL_FEATURES)):
            if w[fi] == 0:
                continue
            d += w[fi] * np.abs(lib["reps"][:, :, fi] - q[None, :, fi]).mean(axis=1)
        nn = np.argpartition(d, k)[:k]
        nn = nn[np.argsort(d[nn])]
        for rank, idx in enumerate(nn):
            row = {p: float(lib["params"][idx, j]) for j, p in enumerate(PARAMS)}
            row.update(group=str(sid), rank=rank, sample_id=int(lib["samples"][idx]),
                       weighted_distance=float(d[idx]))
            rows.append(row)
    if not rows:
        raise ValueError("no spheroid trajectory had >= 2 valid frames")
    return summarise_posterior(pd.DataFrame(rows))


def infer_from_mask_trajectories(masks_path, k: int = 20, strict: bool = False) -> pd.DataFrame:
    """Folder of per-spheroid trajectory subfolders -> trajectory inversion."""
    frames = features_from_folder(masks_path, aggregate=False, strict=strict)
    return infer_from_trajectory(frames, id_col="spheroid_id", k=k)
