# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Real-data walkthrough: Entospletinib dose-response in CLL spheroids
#
# This notebook runs the package end-to-end on eight U-Net-segmented
# CLL spheroid trajectories bundled under `examples/real_masks/`. The
# experimental design is a small slice of the thesis Pass 5b
# dose-response: patient VID1797, four conditions, two replicates for
# the baseline / perturbation arms and four doses for the drug arm.
#
# **What we expect to see**
#
# Entospletinib is a Syk-kinase inhibitor. The IL-2/15/21/CpG cocktail
# is a B-cell activator. In this patient the unstim controls already
# sit in the dewetted regime (high $J_{cc}$, around 30-40), and stim
# *decreases* $J_{cc}$ - presumably by inducing actin remodelling that
# softens cell-cell contacts. The biological prediction: Entospletinib
# at high dose should *block* the Syk-driven softening and rescue
# $J_{cc}$ back up toward the unstim baseline.
#
# **What this notebook shows**
#
# 1. Feature extraction on the bundled masks
# 2. Inversion into CPM parameter posteriors
# 3. Cross-condition shifts on the three identifiable axes
# 4. A simple dose-response plot on the drug arm

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from cll_cpm_inversion import (
    OPERATIONAL_FEATURES, PARAMS,
    features_from_folder,
    infer_from_features,
    load_identifiability,
)

MASKS_DIR = Path("real_masks")
META_CSV  = MASKS_DIR / "metadata.csv"
assert MASKS_DIR.is_dir(), f"missing {MASKS_DIR}"
assert META_CSV.is_file(), f"missing {META_CSV}"

# %% [markdown]
# ## 1. Inspect the bundled masks
#
# Each trajectory folder holds ~19 frames of a single spheroid. We show
# four frames (start / 1/3 / 2/3 / end) from one of the unstim baseline
# trajectories so you can see what the package expects.

# %%
example_well = "VID1797_F1"
frames = sorted((MASKS_DIR / example_well).glob("*.png"))
picks = [frames[0], frames[len(frames)//3], frames[2*len(frames)//3], frames[-1]]

fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
for ax, p in zip(axes, picks):
    ax.imshow(np.array(Image.open(p)), cmap="gray")
    ax.set_title(p.name.split("_")[-1], fontsize=9)
    ax.axis("off")
fig.suptitle(f"{example_well}: 4 of {len(frames)} frames (every 4th hourly snapshot retained)",
             fontsize=11)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Extract features
#
# `features_from_folder` walks the eight subfolders, computes the five
# operational features per frame, and averages them within each
# trajectory.

# %%
features_df = features_from_folder(MASKS_DIR)
features_df.round(2)

# %% [markdown]
# Join the bundled metadata to label each row by condition:

# %%
metadata = pd.read_csv(META_CSV)
features_df = features_df.merge(metadata, on="spheroid_id", how="left")
features_df[["spheroid_id", "condition_label",
             "total_area", "perimeter", "circularity"]].round(2)

# %% [markdown]
# A first sanity check on the raw features: stim wells should have a
# rougher outline (higher perimeter, lower circularity) than unstim
# baselines.

# %%
features_df.groupby("condition_label")[
    ["total_area", "perimeter", "circularity", "solidity"]
].mean().round(2).sort_values("circularity")

# %% [markdown]
# ## 3. Invert into CPM parameter posteriors
#
# The same Sobol-weighted k-NN matcher used in the thesis, against the
# bundled 583-vector synthetic library. The output is one row per
# (spheroid, CPM parameter).

# %%
posterior = infer_from_features(features_df, k=20)
posterior = posterior.merge(metadata, left_on="spheroid_id", right_on="spheroid_id")
posterior[posterior["identifiability"] != "unidentifiable"].round(2).head(15)

# %% [markdown]
# ## 4. Identifiability flags (a property of the library, not your data)

# %%
load_identifiability().round(3)

# %% [markdown]
# Three parameters cross the 0.3 weak-identifiability threshold:
# `cm_adhesion` ($J_{cm}$), `width`, and `contact` ($J_{cc}$). The
# remaining four (`temp`, `lambda`, `contact_no`, `neighbor`) are
# unidentifiable from a single 2D spheroid snapshot - their posteriors
# below should be ignored as point estimates.

# %% [markdown]
# ## 5. Cross-condition shifts on the identifiable axes
#
# Pivot the posterior medians by condition for the three weakly-
# identifiable parameters.

# %%
identifiable = posterior[posterior["identifiability"] != "unidentifiable"]

condition_table = (identifiable
    .groupby(["condition_label", "parameter"])["median"]
    .median().unstack("parameter")
    .reindex([
        "unstim_baseline_rep1", "unstim_baseline_rep2",
        "stim_control_rep1", "stim_control_rep2",
        "stim_plus_drug_100nM", "stim_plus_drug_10nM",
        "stim_plus_drug_1nM", "stim_plus_drug_0.1nM",
    ])
    .round(2))
condition_table

# %% [markdown]
# Read the table top-to-bottom:
#
# - **Unstim baseline (F1/F2):** $J_{cc}$ is high (around 30); the
#   spheroid is already in the dewetted regime at rest.
# - **Stim control (F5/F6):** stim *decreases* $J_{cc}$ to ~19-23
#   (softens cell-cell contacts) and pulls $J_{cm}$ down toward the
#   wetting boundary.
# - **Stim + Entospletinib 100 nM (A5):** drug at high dose pushes
#   $J_{cc}$ back up to ~38 - past the unstim baseline. This is the
#   Syk-blockade rescue signature: the drug prevents the stim-induced
#   contact softening.
# - **Lower doses (A6 -> A8):** $J_{cc}$ stays high at 10 nM (~40) and
#   collapses back to stim-control levels at 1 nM and 0.1 nM - the
#   rescue attenuates between 10 nM and 1 nM.
#
# Width is essentially unchanged across all conditions (~25), because
# total area is preserved on the time scale of these experiments.

# %% [markdown]
# ## 6. Posterior interval plot
#
# Show the 5th-95th percentile interval per condition for the three
# identifiable parameters. This is the form that goes into thesis
# Figure 4.13.

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
identifiable_params = ["width", "contact", "cm_adhesion"]
labels = {"width": r"width $w$",
          "contact": r"$J_{cc}$",
          "cm_adhesion": r"$J_{cm}$"}

cond_order = [
    "unstim_baseline_rep1", "unstim_baseline_rep2",
    "stim_control_rep1", "stim_control_rep2",
    "stim_plus_drug_100nM", "stim_plus_drug_10nM",
    "stim_plus_drug_1nM", "stim_plus_drug_0.1nM",
]
colors = {
    "unstim_baseline_rep1":   "#5c9ead", "unstim_baseline_rep2":   "#5c9ead",
    "stim_control_rep1":      "#c75b50", "stim_control_rep2":      "#c75b50",
    "stim_plus_drug_100nM":   "#3a6940", "stim_plus_drug_10nM":    "#6fa570",
    "stim_plus_drug_1nM":     "#9bc89c", "stim_plus_drug_0.1nM":   "#cfe3cf",
}

for ax, p in zip(axes, identifiable_params):
    sub = identifiable[identifiable["parameter"] == p].set_index("spheroid_id")
    for i, cond in enumerate(cond_order):
        cond_rows = posterior[(posterior["condition_label"] == cond) &
                              (posterior["parameter"] == p)]
        if len(cond_rows) == 0:
            continue
        sid = cond_rows["spheroid_id"].iloc[0]
        row = sub.loc[sid]
        ax.plot([i, i], [row["q05"], row["q95"]], color=colors[cond], lw=2)
        ax.scatter([i], [row["median"]], color=colors[cond], s=40, zorder=3)
    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels(cond_order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(labels[p])
    ax.set_title(labels[p])
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Posterior 90% intervals per condition (real VID1797 data)",
             fontsize=12)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Dose-response on the drug arm
#
# Slice to the four Entospletinib + stim wells and plot $J_{cc}$
# (cell-cell adhesion) against dose. The expected curve is monotonic
# rescue: $J_{cc}$ starts at the stim-control level at the lowest
# dose and climbs back toward (or past) the unstim baseline as dose
# increases.

# %%
drug = posterior[(posterior["drug"] == "Entospletinib") &
                 (posterior["parameter"] == "contact")].copy()
drug = drug.sort_values("concentration_nM", ascending=False)

# Reference lines: unstim baseline and stim control medians
unstim_jcc = posterior[(posterior["condition_label"].str.startswith("unstim")) &
                       (posterior["parameter"] == "contact")]["median"].median()
stim_jcc   = posterior[(posterior["condition_label"].str.startswith("stim_control")) &
                       (posterior["parameter"] == "contact")]["median"].median()

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.errorbar(drug["concentration_nM"], drug["median"],
            yerr=[drug["median"] - drug["q05"], drug["q95"] - drug["median"]],
            marker="o", linewidth=1.5, capsize=4, color="#3a6940",
            label="Entospletinib + stim")
ax.axhline(unstim_jcc, color="#5c9ead", linestyle="--",
           label=f"unstim baseline median ({unstim_jcc:.1f})")
ax.axhline(stim_jcc, color="#c75b50", linestyle="--",
           label=f"stim control median ({stim_jcc:.1f})")
ax.set_xscale("log")
ax.set_xlabel("Entospletinib (nM)")
ax.set_ylabel(r"posterior median $J_{cc}$")
ax.set_title(r"Dose-response in $J_{cc}$ space (VID1797)")
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)
plt.show()

# %% [markdown]
# ## 8. What you should take away
#
# - The same package + same library + same matcher used in the thesis
#   reproduces, on real masks, the Pass 5b finding: stim *softens*
#   $J_{cc}$ (drops by ~10), and Entospletinib at >= 10 nM *rescues*
#   $J_{cc}$ back up toward (or past) the unstim baseline. The rescue
#   attenuates between 10 nM and 1 nM.
# - The unidentifiable parameters in the posterior remain at their
#   prior expectation regardless of condition - this is the practical
#   signature of unidentifiability and is exactly what the
#   identifiability flag is for.
# - Absolute values of $J_{cc}$ and $J_{cm}$ are extrapolated outside
#   the bundled synthetic library's nearest-neighbour envelope (see
#   `docs/METHODS.md` section 6). Treat them as "closest synthetic
#   regime", and read **shifts** between conditions, not absolute
#   point values.
