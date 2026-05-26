"""cll_cpm_inversion: invert spheroid morphology into CPM parameter posteriors.

High-level API:
    infer_from_masks(masks_path, k=20)          -> posterior DataFrame
    infer_from_features(features_df, k=20)      -> posterior DataFrame

Low-level API:
    features.extract_features_from_mask(mask)   -> dict
    features.features_from_folder(folder)       -> DataFrame
    invert.invert_groups(obs_by_group, ...)     -> DataFrame
    invert.load_synthetic_library()             -> DataFrame
    invert.load_sobol_indices()                 -> DataFrame
    invert.load_identifiability()               -> DataFrame

Constants:
    OPERATIONAL_FEATURES, PARAMS, IDENTIFIABILITY_THRESHOLDS
"""
from .features import (
    OPERATIONAL_FEATURES,
    extract_features_from_mask,
    features_from_folder,
)
from .invert import (
    PARAMS,
    IDENTIFIABILITY_THRESHOLDS,
    invert_groups,
    invert_observations,
    load_synthetic_library,
    load_sobol_indices,
    load_identifiability,
    feature_weights_from_sobol,
    summarise_posterior,
)
from .pipeline import infer_from_masks, infer_from_features, infer_from_coords
from .coords_io import features_from_coords, read_coords

__version__ = "0.1.0"
__all__ = [
    "infer_from_masks",
    "infer_from_features",
    "infer_from_coords",
    "features_from_coords",
    "read_coords",
    "extract_features_from_mask",
    "features_from_folder",
    "invert_observations",
    "invert_groups",
    "summarise_posterior",
    "feature_weights_from_sobol",
    "load_synthetic_library",
    "load_sobol_indices",
    "load_identifiability",
    "OPERATIONAL_FEATURES",
    "PARAMS",
    "IDENTIFIABILITY_THRESHOLDS",
]
