# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (uva-aml-2025)
#     language: python
#     name: uva-aml-2025
# ---

# %% [markdown]
# # Inversion-only: bypass the feature-extraction step
#
# If you already have a CSV of morphology features (e.g. from a
# different segmenter or a different feature library), feed it
# directly into `infer_from_features`. Required columns:
#
# - `spheroid_id`
# - `total_area`
# - `equivalent_diameter`
# - `solidity`
# - `perimeter`
# - `circularity`
#
# Any other columns are passed through. One row per spheroid (or per
# frame; the matcher does not distinguish - if you pass frame-resolved
# rows the matcher treats each row as an independent observation and
# the posterior summary will be over all rows for that spheroid_id).

# %%
import pandas as pd

from cll_cpm_inversion import (
    OPERATIONAL_FEATURES, PARAMS,
    infer_from_features,
    load_synthetic_library,
    load_identifiability,
    load_sobol_indices,
    feature_weights_from_sobol,
)

# %% [markdown]
# ## Worked example: feed three library samples back in
#
# A useful sanity check: a synthetic vector from the library should
# match itself at rank 0 and yield a tight posterior around the truth.

# %%
lib = load_synthetic_library()
picks = lib.iloc[[10, 200, 400]].copy()
picks.index = ["lib_010", "lib_200", "lib_400"]
features_df = picks[OPERATIONAL_FEATURES].reset_index().rename(
    columns={"index": "spheroid_id"})
features_df

# %%
summary = infer_from_features(features_df, k=20)
summary.round(3)

# %% [markdown]
# Compare posterior median to ground truth:

# %%
truth = picks[PARAMS].reset_index().rename(columns={"index": "spheroid_id"})
truth_long = truth.melt(id_vars="spheroid_id",
                        var_name="parameter", value_name="truth")
compare = summary.merge(truth_long, on=["spheroid_id", "parameter"])
compare[["spheroid_id", "parameter", "truth", "median",
         "q05", "q95", "identifiability"]].round(2)

# %% [markdown]
# Notice that even when feeding the library back into itself, the
# unidentifiable parameters (`temp`, `lambda`, `contact_no`,
# `neighbor`) are not recovered to their true values - the matcher
# averages over the 20 most morphologically-similar synthetic samples,
# and those samples have very different values of the unidentifiable
# parameters. This is the practical signature of unidentifiability:
# the feature-to-parameter mapping does not constrain those
# parameters.

# %% [markdown]
# ## Inspecting the matcher internals
#
# The Sobol indices and per-feature weights that drive the matcher
# are all loadable.

# %%
load_sobol_indices().head()

# %%
weights = feature_weights_from_sobol()
pd.Series(weights, name="weight").to_frame().round(4)

# %% [markdown]
# `circularity` carries the most weight because its mean total-order
# Sobol index across the 7 parameters is the largest in this library.
# This is what promotes $J_{cc}$ from unidentifiable (under uniform
# weights, R^2 < 0.3) to weakly identifiable.
