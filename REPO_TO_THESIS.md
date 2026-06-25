# Repository to thesis map

Where each part of this repository is used in the thesis:

> D.D. Zair, *Inferring Cellular Potts Model Parameters from Chronic Lymphocytic
> Leukaemia Tumour Morphology*, MSc thesis, University of Amsterdam, 2026.

This package is the **inference engine and its bundled data**: it takes spheroid
morphology (masks, a feature table, or a coordinate CSV) and returns CPM
parameter posteriors with identifiability flags. The thesis figure-generation
scripts are **not** part of this repository; they live in the thesis working
tree and consume this package's outputs. The thesis appendix cites this repo as
the reproducibility source (Appendix "Hardware, software, and reproducibility",
`app:reproducibility`).

## Code modules

| Repo path | What it is | Thesis section / label |
|---|---|---|
| `cll_cpm_inversion/features.py` | Extracts the five operational morphology features (area, equivalent diameter, solidity, perimeter, circularity) from a mask. | Methodology, "Segmentation and morphology extraction" (`sec:method_morphology_extraction`); RQ1 (`rq:1`). |
| `cll_cpm_inversion/trajectory.py` | **Primary matcher**: tau-registered trajectory matching against the bundled library trajectories (`infer_from_trajectory`, `infer_from_mask_trajectories`). | Experimental setup, "Matching" (`sec:setup_matching`); matching schematic (`fig:matching_schematic`); RQ3 (`rq:3`). |
| `cll_cpm_inversion/invert.py` | k-NN posterior matching, the static end-state matcher (`invert_observations`), LOO identifiability flags, and `summarise_posterior`. | Methodology, "Simulation-based parameter inference"; identifiability table (`tab:rq2_1_identifiability`); RQ2 (`rq:2`, `rq:2.1`). |
| `cll_cpm_inversion/pipeline.py` | Top-level entry points (`infer_from_masks`, `infer_from_features`, `infer_from_coords`). | Methodology pipeline overview. |
| `cll_cpm_inversion/coords_io.py` | Coordinate-CSV (`cell_id, x, y`) input adapter, rasterised to a mask. | Methodology (portable input format). |
| `cll_cpm_inversion/cli.py` | `cll-invert` command-line interface. | Reproducibility appendix (`app:reproducibility`). |

## Bundled data (`cll_cpm_inversion/data/`)

| Repo path | What it is | Thesis section / label |
|---|---|---|
| `synthetic_library.csv` | The fixed 1105-vector synthetic library (7-parameter Saltelli design); the matching contract. | Experimental setup, Sobol design (`sec:setup_sobol`); the "fixed simulation library" referenced throughout. |
| `synthetic_library_trajectories.csv.gz` | Per-frame trajectories for every library vector; used by the primary trajectory matcher. | Trajectory matching (`sec:setup_matching`). |
| `identifiability_loo.csv` | Leave-one-out R^2 per parameter; the source of the identifiable / weakly / unidentifiable flags. | Identifiability results (`tab:rq2_1_identifiability`); RQ2.1 (`rq:2.1`). |
| `sobol_indices.csv` | Sobol sensitivity indices from the XGBoost surrogate; the source of the per-feature matching weights. | Sobol setup (`sec:setup_sobol`); Sobol estimators / bars and surrogate CV appendices (`app:sobol_estimators`, `app:sobol_bars`, `app:surrogate_cv`, `tab:surrogate_benchmark`). |
| `synthetic_library_eda_report.md` | Provenance and EDA of the bundled library. | Library construction / data provenance. |

## Worked examples (`examples/`)

| Repo path | What it is | Thesis section / label |
|---|---|---|
| `examples/real_masks/` + `example_trajectories.csv` + `03_real_data_walkthrough.ipynb` | Eight to twenty VID1797 spheroids, the Entospletinib dose ladder plus controls. Reproduces the Pass 5b finding: stimulation suppresses cell-cell adhesion (`contact`, J_cc) and a high dose restores it. | Results, stimulation-response reproducibility (`rq:3.1`). |
| `examples/drug_panel_demo/` | Four U-Net-segmented trajectories, one per mechanism class (BTKi, PI3Kdelta, NF-kB, CXCR4). "Runs across mechanisms" demo (unstimulated, single-dose; not a rescue result). | Drug panel / lookup (`tab:drug_lookup`). |
| `examples/01_quickstart.ipynb`, `02_inversion_only.ipynb` | Synthetic walk-throughs of the inference API. | Methodology pipeline overview. |
| `demo.py` | One-command demo (`python demo.py`) over the bundled real masks; no network or external data. | Reproducibility appendix (`app:reproducibility`). |

## Documentation (`docs/`)

| Repo path | What it covers |
|---|---|
| `docs/METHODS.md` | Matching method, surrogate, and the upstream CompuCell3D simulator (DOI). |
| `docs/PARAMETERS.md` | The seven CPM parameters and their symbols (J_cm, w, J_cc, lambda_V, T, r_ct, n_ord). |
| `docs/DATA_PREPARATION.md` | Accepted mask formats and folder layouts. |
| `docs/CONTRIBUTING.md` | Development and test instructions. |

Note: label names above are LaTeX `\label{}` anchors in the thesis source; cite
the rendered section/table/figure numbers in the final document.
