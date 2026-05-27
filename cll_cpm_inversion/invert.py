"""Sobol-weighted k-NN inversion against the bundled Saltelli library.

The bundled CSV `data/synthetic_library.csv` is a joined table of 1105
nan-free Saltelli vectors: each row is one synthetic spheroid with its
7 CPM parameters and 5 (+ eccentricity) replicate-averaged final-MCS
features. The same per-feature Sobol weights and identifiability R^2 used
in the thesis are loaded from `data/sobol_indices.csv` and
`data/identifiability_loo.csv`.
"""
from __future__ import annotations

from importlib import resources
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .features import OPERATIONAL_FEATURES

PARAMS = [
    "width", "temp", "lambda", "contact", "cm_adhesion",
    "contact_no", "neighbor",
]

# Operational identifiability thresholds (Section 4 of the thesis): the
# leave-one-out R^2 on the bundled library is compared against these.
IDENTIFIABILITY_THRESHOLDS = {"identifiable": 0.7, "weak": 0.3}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _data_path(name: str):
    return resources.files("cll_cpm_inversion").joinpath("data", name)


def load_synthetic_library() -> pd.DataFrame:
    """1105 rows of (PARAMS + features). Index = sample_id."""
    path = _data_path("synthetic_library.csv")
    df = pd.read_csv(path)
    if "sample_id" in df.columns:
        df = df.set_index("sample_id")
    return df


def load_sobol_indices() -> pd.DataFrame:
    """Long table: feature, parameter, S1, ST (+ confidence bands).

    Used to compute per-feature weights for the matcher.
    """
    return pd.read_csv(_data_path("sobol_indices.csv"))


def load_identifiability() -> pd.DataFrame:
    """Per-parameter leave-one-out R^2 (and Pearson r) from the bundled library."""
    return pd.read_csv(_data_path("identifiability_loo.csv"))


# ---------------------------------------------------------------------------
# Feature weights from Sobol total-order indices
# ---------------------------------------------------------------------------

def feature_weights_from_sobol(sobol_df: pd.DataFrame | None = None,
                               features: Iterable[str] = OPERATIONAL_FEATURES,
                               mode: str = "ST_mean") -> dict[str, float]:
    """Per-feature weight for the matching metric.

    A feature with high mean total-order Sobol index (ST) across the 7
    parameters carries more identifiability signal and gets a larger
    weight. The weights are normalised to sum to 1.
    """
    if sobol_df is None:
        sobol_df = load_sobol_indices()
    if mode == "ST_mean":
        w = (sobol_df[sobol_df["feature"].isin(features)]
             .groupby("feature")["ST"].mean())
    elif mode == "ST_max":
        w = (sobol_df[sobol_df["feature"].isin(features)]
             .groupby("feature")["ST"].max())
    elif mode == "uniform":
        w = pd.Series(1.0, index=list(features))
    else:
        raise ValueError(f"unknown mode {mode!r}")
    w = w.reindex(list(features)).fillna(0.0)
    total = float(w.sum())
    if total <= 0:
        return {f: 1.0 / len(list(features)) for f in features}
    return {f: float(v) / total for f, v in w.items()}


# ---------------------------------------------------------------------------
# Identifiability flag from LOO R^2
# ---------------------------------------------------------------------------

def identifiability_flag(r2: float) -> str:
    if r2 >= IDENTIFIABILITY_THRESHOLDS["identifiable"]:
        return "identifiable"
    if r2 >= IDENTIFIABILITY_THRESHOLDS["weak"]:
        return "weakly identifiable"
    return "unidentifiable"


# ---------------------------------------------------------------------------
# k-NN inversion
# ---------------------------------------------------------------------------

def _invert_one_group(observations: pd.DataFrame,
                      sim_table: pd.DataFrame,
                      features: list[str],
                      sample_ids: np.ndarray,
                      sim_params: np.ndarray,
                      scaler: StandardScaler,
                      sim_z_w: np.ndarray,
                      sqrt_w: np.ndarray,
                      k: int,
                      group_label: str) -> pd.DataFrame:
    obs_z = scaler.transform(observations[features].values)
    obs_z_w = obs_z * sqrt_w

    diffs = obs_z_w[:, None, :] - sim_z_w[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))

    rows = []
    for oi in range(observations.shape[0]):
        idx_sorted = np.argsort(dists[oi])[:k]
        for rank, si in enumerate(idx_sorted):
            row = {p: float(sim_params[si, j]) for j, p in enumerate(PARAMS)}
            row.update({
                "group":             group_label,
                "obs_idx":           int(oi),
                "rank":              int(rank),
                "weighted_distance": float(dists[oi, si]),
                "sample_id":         int(sample_ids[si]),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def invert_groups(observations_by_group: Mapping[str, pd.DataFrame],
                  sim_table: pd.DataFrame | None = None,
                  weights: dict[str, float] | None = None,
                  k: int = 20) -> pd.DataFrame:
    """Invert one or more groups of observations against the synthetic library.

    Parameters
    ----------
    observations_by_group : dict[group_label -> DataFrame]
        Each DataFrame has n_obs rows, columns include OPERATIONAL_FEATURES.
    sim_table : DataFrame indexed by sample_id with PARAMS + features.
        Defaults to the bundled library.
    weights : per-feature weights. Defaults to ST_mean weights from the
        bundled Sobol indices.
    k : top-k matches per observation contribute to the group posterior.
        Default 20.

    Returns
    -------
    DataFrame, one row per (observation, rank). Columns = PARAMS + meta.
    """
    if sim_table is None:
        sim_table = load_synthetic_library()
    if weights is None:
        weights = feature_weights_from_sobol(features=OPERATIONAL_FEATURES)

    features = list(weights.keys())
    w_vec = np.array([weights[f] for f in features], dtype=float)
    w_vec = w_vec / w_vec.sum()

    scaler = StandardScaler().fit(sim_table[features].values)
    sim_z = scaler.transform(sim_table[features].values)
    sqrt_w = np.sqrt(w_vec)[None, :]
    sim_z_w = sim_z * sqrt_w
    sample_ids = sim_table.index.to_numpy()
    sim_params = sim_table[PARAMS].values

    chunks = []
    for label, obs in observations_by_group.items():
        if len(obs) == 0:
            continue
        chunks.append(_invert_one_group(
            obs, sim_table, features, sample_ids, sim_params,
            scaler, sim_z_w, sqrt_w, k, str(label),
        ))
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def invert_observations(features_df: pd.DataFrame,
                        id_col: str = "spheroid_id",
                        k: int = 20,
                        sim_table: pd.DataFrame | None = None,
                        weights: dict[str, float] | None = None) -> pd.DataFrame:
    """One spheroid per group: thin wrapper over `invert_groups`.

    Parameters
    ----------
    features_df : DataFrame
        Must contain `id_col` and OPERATIONAL_FEATURES.
    id_col : column to use as spheroid identifier.
    k : top-k matches per observation.

    Returns
    -------
    Long posterior DataFrame; `group` column == spheroid id.
    """
    missing = [f for f in OPERATIONAL_FEATURES if f not in features_df.columns]
    if missing:
        raise ValueError(f"features_df missing required columns: {missing}")
    obs_by_id = {str(sid):
                 sub[OPERATIONAL_FEATURES].reset_index(drop=True)
                 for sid, sub in features_df.groupby(id_col, sort=False)}
    return invert_groups(obs_by_id, sim_table=sim_table, weights=weights, k=k)


# ---------------------------------------------------------------------------
# Posterior summary
# ---------------------------------------------------------------------------

def summarise_posterior(posterior: pd.DataFrame,
                        loo: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per (group, parameter): median, 5th/95th percentile, identifiability flag.

    The 5th-95th percentile interval is the empirical 90% interval used in
    the thesis. The flag is taken from the bundled LOO R^2 table.
    """
    if loo is None:
        loo = load_identifiability()
    r2_by_param = dict(zip(loo["param"], loo["R2"]))

    rows = []
    for (group, sub) in posterior.groupby("group", sort=False):
        for p in PARAMS:
            vals = sub[p].dropna().values
            if len(vals) == 0:
                continue
            r2 = r2_by_param.get(p, float("nan"))
            rows.append({
                "spheroid_id":     group,
                "parameter":       p,
                "median":          float(np.median(vals)),
                "q05":             float(np.percentile(vals, 5)),
                "q95":             float(np.percentile(vals, 95)),
                "q25":             float(np.percentile(vals, 25)),
                "q75":             float(np.percentile(vals, 75)),
                "n_matches":       int(len(vals)),
                "loo_r2":          float(r2) if r2 == r2 else float("nan"),
                "identifiability": identifiability_flag(r2) if r2 == r2 else "unknown",
            })
    return pd.DataFrame(rows)
