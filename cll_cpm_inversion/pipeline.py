"""End-to-end orchestration: masks (or features) in, posterior summary out."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from .features import OPERATIONAL_FEATURES, features_from_folder
from .invert import invert_observations, summarise_posterior

PathLike = Union[str, Path]


def infer_from_masks(masks_path: PathLike,
                     k: int = 20,
                     aggregate_trajectories: bool = True,
                     strict: bool = False,
                     return_features: bool = False,
                     ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Extract features from a folder of masks and return posterior summary.

    Parameters
    ----------
    masks_path : path to folder. Layout = flat (one mask file per
        spheroid) or nested (one subfolder per spheroid with frames inside).
    k : top-k matches per spheroid. Default 20.
    aggregate_trajectories : if a trajectory folder is given, average
        per-frame features into one row per spheroid (default). Set to
        False to invert each frame independently.
    strict : if True, raise ValueError on any frame containing more than
        one large connected component (defends against silent loss of a
        second spheroid). Default False (lenient: keep the largest, drop
        the rest).
    return_features : if True, return (posterior_summary, features_df).

    Returns
    -------
    DataFrame, one row per (spheroid, parameter), with columns
    [spheroid_id, parameter, median, q05, q95, q25, q75, n_matches,
     loo_r2, identifiability].
    """
    features_df = features_from_folder(masks_path,
                                       aggregate=aggregate_trajectories,
                                       strict=strict)
    nan_rows = features_df[OPERATIONAL_FEATURES].isna().any(axis=1)
    if nan_rows.any():
        bad = features_df.loc[nan_rows, "spheroid_id"].tolist()
        features_df = features_df.loc[~nan_rows].reset_index(drop=True)
        print(f"WARN: dropped {len(bad)} spheroids with NaN features: {bad}")

    posterior = invert_observations(features_df, k=k)
    summary = summarise_posterior(posterior)
    if return_features:
        return summary, features_df
    return summary


def infer_from_features(features_df: pd.DataFrame,
                        id_col: str = "spheroid_id",
                        k: int = 20) -> pd.DataFrame:
    """Skip feature extraction: invert a pre-built feature table directly.

    Parameters
    ----------
    features_df : DataFrame with `id_col` and the five OPERATIONAL_FEATURES.
    id_col : spheroid identifier column. Default 'spheroid_id'.
    k : top-k matches.

    Returns
    -------
    Same as `infer_from_masks`.
    """
    posterior = invert_observations(features_df, id_col=id_col, k=k)
    return summarise_posterior(posterior)


def infer_from_coords(coords_csv: PathLike,
                      k: int = 20,
                      strict: bool = False) -> pd.DataFrame:
    """Invert a ``(cell_id, x, y)`` coordinate CSV into CPM parameter posteriors.

    Each distinct id is one spheroid; its rows are the foreground pixels. The
    points are rasterised to a mask and run through the same feature extraction
    as the image path, then matched against the synthetic library.

    Returns
    -------
    Same posterior summary as ``infer_from_masks`` / ``infer_from_features``.
    """
    from .coords_io import features_from_coords

    features = features_from_coords(coords_csv, strict=strict)
    return infer_from_features(features, k=k)
