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
# # cll-cpm-inversion: quickstart
#
# This notebook walks through the full pipeline end-to-end on a small
# set of synthetic mask images. The same code works on real segmented
# spheroid masks.
#
# **Pipeline**
#
# 1. Generate a folder of mask files (you would normally already have
#    these from your segmenter; here we create synthetic disks so the
#    notebook is self-contained).
# 2. Extract the five morphology features from each mask.
# 3. Invert into CPM parameter posteriors.
# 4. Read the per-parameter identifiability flags and the posterior
#    median + 90% interval.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from cll_cpm_inversion import (
    OPERATIONAL_FEATURES,
    PARAMS,
    extract_features_from_mask,
    features_from_folder,
    infer_from_features,
    infer_from_masks,
    load_identifiability,
)

# %% [markdown]
# ## 1. Create a synthetic mask folder
#
# Replace this step with your own segmented spheroid masks. The
# package accepts `.jpg`, `.png`, `.tif`, and `.tiff` files. Pixels
# with value > 0 are treated as spheroid.

# %%
WORK = Path("/tmp/cll_quickstart")
WORK.mkdir(exist_ok=True)

def disk(radius: int, size: int = 200) -> np.ndarray:
    y, x = np.ogrid[:size, :size]
    cy = cx = size // 2
    return ((y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2).astype(np.uint8) * 255

# Layout: one subfolder per spheroid, frames inside
for spheroid_id, radii in [
    ("W001_unstim", [20, 22, 24, 26]),
    ("W002_unstim", [22, 25, 28, 30]),
    ("W003_unstim", [18, 20, 21, 22]),
    ("W004_stim",   [30, 35, 40, 45]),
    ("W005_stim",   [28, 33, 38, 42]),
]:
    sd = WORK / spheroid_id
    sd.mkdir(exist_ok=True)
    for frame, r in enumerate(radii):
        Image.fromarray(disk(r)).save(sd / f"frame_{frame:02d}.tif")

print("Created", WORK, "with", len(list(WORK.iterdir())), "trajectory folders.")

# %% [markdown]
# ## 2. Extract features
#
# `features_from_folder` walks the folder. With one subfolder per
# spheroid (the layout above), it averages the per-frame features into
# one representative vector per spheroid. Set `aggregate=False` to keep
# one row per (spheroid, frame).

# %%
features_df = features_from_folder(WORK)
features_df.round(3)

# %% [markdown]
# ## 3. Run the inversion
#
# The simplest call: `infer_from_masks(folder)` runs steps 2 and 3
# together. Here we already have the features, so we use
# `infer_from_features` directly.

# %%
posterior = infer_from_features(features_df, k=20)
posterior.round(3).head(20)

# %% [markdown]
# ## 4. Read the identifiability flags
#
# The library's leave-one-out R^2 tells us how recoverable each
# parameter is from 2D morphology *at all*. This is a property of the
# library, not of your spheroid - it bounds how seriously to read each
# row of the posterior.

# %%
load_identifiability().round(3)

# %% [markdown]
# Three of the seven parameters cross the 0.3 threshold and are flagged
# `weakly identifiable`: width, contact ($J_{cc}$), and cm_adhesion
# ($J_{cm}$). The other four are `unidentifiable` from a single 2D
# spheroid snapshot and their posterior values should be read as the
# prior expectation, not as point estimates.

# %% [markdown]
# ## 5. Pivot for a clean report
#
# Most users will want a wide table: one row per spheroid, one column
# per identifiable parameter.

# %%
identifiable = posterior[posterior["identifiability"] != "unidentifiable"]
wide_median = identifiable.pivot(index="spheroid_id",
                                 columns="parameter",
                                 values="median").round(2)
wide_median

# %% [markdown]
# ## 6. Comparing conditions
#
# When you have multiple conditions (stim vs unstim, drug vs vehicle,
# patient A vs patient B), the *defensible read* is the cross-condition
# *shift* on the identifiable axes, not absolute parameter values
# (~90% of real CLL spheroids extrapolate the bundled library, see
# `docs/METHODS.md` section 6).

# %%
features_df["condition"] = features_df["spheroid_id"].str.split("_").str[-1]
features_df.groupby("condition")[OPERATIONAL_FEATURES].mean().round(2)

# %%
posterior["condition"] = posterior["spheroid_id"].str.split("_").str[-1]
by_condition = (posterior[posterior["identifiability"] != "unidentifiable"]
                .groupby(["condition", "parameter"])["median"]
                .median().unstack("parameter").round(2))
by_condition

# %% [markdown]
# ## 7. CLI equivalent
#
# Everything above can be done from the shell:
#
# ```bash
# cll-invert /tmp/cll_quickstart --out /tmp/posteriors.csv
# ```
#
# The CLI writes the same long posterior summary CSV and prints the
# pivoted table of identifiable parameter medians.
