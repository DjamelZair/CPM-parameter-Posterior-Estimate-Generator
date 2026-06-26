# cll-cpm-inversion

Invert segmented spheroid morphology into Cellular Potts Model (CPM)
parameter posteriors, with per-parameter identifiability flags.

Input is either a folder of segmented spheroid masks (JPG / TIFF / PNG), a
pre-built feature CSV, or a **coordinate CSV** (`cell_id, x, y`, one row per
foreground pixel). Given any of these, this package:

1. extracts the five operational morphology features (area, equivalent
   diameter, solidity, perimeter, circularity);
2. matches each observation against a bundled 1105-vector synthetic library
   sampled with a 7-parameter Saltelli design;
3. returns a posterior distribution over CPM parameters (median + 5th / 95th
   percentiles) using Sobol-weighted k-nearest-neighbour matching. The
   **primary matcher is trajectory-based** (`infer_from_trajectory`): it
   compares the whole time course against the bundled library *trajectories*
   on a shared phase axis (tau-registration). A **static end-state matcher**
   is retained for comparison; the two are complementary (trajectory matching
   is better for identifiability, end-state is the more sensitive probe of a
   drug-induced settled state);
4. flags each parameter as **identifiable**, **weakly identifiable**, or
   **unidentifiable** based on the leave-one-out R^2 of the bundled library.

## Install

```bash
pip install git+https://github.com/DjamelZair/CPM-parameter-Posterior-Estimate-Generator.git
```

or, from a clone:

```bash
git clone https://github.com/DjamelZair/CPM-parameter-Posterior-Estimate-Generator.git
cd CPM-parameter-Posterior-Estimate-Generator
pip install -e .
```

## Quickstart (CLI)

```bash
cll-invert /path/to/masks/  --out posteriors.csv              # end-state matcher
cll-invert /path/to/masks/  --trajectory  --out posteriors.csv  # trajectory matcher (primary)
```

For the **trajectory matcher**, point at a Layout-B folder (one subfolder per
spheroid, frames inside); each trajectory is tau-registered and matched against
the bundled library trajectories:

```python
import cll_cpm_inversion as ci
posteriors = ci.infer_from_mask_trajectories("/path/to/masks_root/", k=20)
# or from a per-frame feature table:
posteriors = ci.infer_from_trajectory(frames_df, id_col="spheroid_id", frame_col="frame")
```

You can also invert a **coordinate CSV** (`cell_id, x, y`), the portable input
format: each distinct `cell_id` is one spheroid and its rows are the foreground
pixels. Column aliases `spheroid_id`/`id`/`label`, `col`/`cx`, `row`/`cy` are
accepted, and the delimiter (`,` / `;` / tab) is auto-detected.

```bash
cll-invert --coords-csv examples/example_coords.csv  --out posteriors.csv
```

```python
import cll_cpm_inversion as ci
posteriors = ci.infer_from_coords("examples/example_coords.csv", k=20)
```

A coordinate CSV is rasterised back to a mask and run through the same feature
extraction as the image path, so it yields identical features to the equivalent
mask (verified in `tests/test_coords_io.py`).

### CSV-only demo: trajectory matching from a feature table

If your data is already a table of per-frame shape features (no images), feed it
straight to the primary (trajectory) matcher. The CSV needs one row per frame
with columns `spheroid_id, frame, total_area, equivalent_diameter, solidity,
perimeter, circularity`; rows sharing a `spheroid_id` form one trajectory,
ordered by `frame`:

```bash
cll-invert --features-csv examples/example_trajectories.csv \
           --trajectory --frame-col frame --out posteriors.csv
```

`examples/example_trajectories.csv` bundles 20 real VID1797 spheroids (an
Entospletinib dose ladder plus controls). Running it reproduces the thesis Pass
5b finding directly from the CSV: stimulation suppresses inferred cell-cell
adhesion (`contact`, J_cc), and a high drug dose restores it.

| well    | condition                    | J_cc (`contact`) median |
|---------|------------------------------|-------------------------|
| F1 / F2 | unstimulated control         | 27.0 / 38.5             |
| F5 / F6 | stimulated control           | 16.1 / 16.1             |
| A8      | stim + 0.1 nM Entospletinib  | 16.1                    |
| A5      | stim + 100 nM Entospletinib  | 35.6                    |

The same call in Python:

```python
import pandas as pd, cll_cpm_inversion as ci
frames = pd.read_csv("examples/example_trajectories.csv")
posteriors = ci.infer_from_trajectory(frames, id_col="spheroid_id", frame_col="frame")
```

`/path/to/masks/` is a folder where each subfolder is one spheroid
trajectory (one mask file per frame), or a flat folder of single-frame
masks. Files can be `.jpg`, `.png`, or `.tif/.tiff`. White or non-zero
pixels are treated as the spheroid; black or zero pixels as background.

For full details on accepted formats, folder layouts, and the rules
collaborators should follow when sending you masks, see
`docs/DATA_PREPARATION.md`. Use `--strict` to error out (instead of
silently keeping the largest) when an image contains more than one
large connected component.

The output CSV has one row per (spheroid, parameter):

| spheroid_id | parameter   | median | q05  | q95  | identifiability       |
|-------------|-------------|--------|------|------|----------------------|
| W001        | width       | 17.0   | 9.0  | 25.0 | weakly identifiable  |
| W001        | cm_adhesion | 17.4   | 11.2 | 23.1 | weakly identifiable  |
| W001        | contact     | 38.7   | 19.8 | 49.1 | weakly identifiable  |
| W001        | lambda      | 12.5   |  2.3 | 19.7 | unidentifiable       |
| ...         | ...         | ...    | ...  | ...  | ...                  |

It also reports `prior_low`, `prior_high`, and `interval_frac_of_prior` per
parameter, so the `q05-q95` interval can be read as a **narrowed calibration
target**: e.g. an `interval_frac_of_prior` of 0.40 means the posterior pins
that parameter to 40% of its prior sweep range. Identifiable parameters narrow
substantially (~0.35-0.6); unidentifiable ones stay near 1.0 (no narrowing),
which is the signal not to calibrate on them.

## Quickstart (Python)

```python
from cll_cpm_inversion import infer_from_masks

posterior = infer_from_masks("/path/to/masks/")
posterior.to_csv("posteriors.csv", index=False)

print(posterior[posterior.identifiability != "unidentifiable"]
      .pivot(index="spheroid_id", columns="parameter", values="median"))
```

See `examples/01_quickstart.ipynb` for a synthetic walk-through. For a
**real-data** walk-through on eight U-Net-segmented CLL spheroid
trajectories (bundled under `examples/real_masks/`), see
`examples/03_real_data_walkthrough.ipynb`. It reproduces the Pass 5b
finding from the thesis: Entospletinib at high dose rescues $J_{cc}$
back to the unstim baseline.

## What this package gives you

| Parameter   | Symbol     | LOO R^2 | Flag                 |
|-------------|-----------|---------|----------------------|
| cm_adhesion | $J_{cm}$  | 0.62    | weakly identifiable  |
| width       | $w$       | 0.54    | weakly identifiable  |
| contact     | $J_{cc}$  | 0.38    | weakly identifiable  |
| lambda      | $\lambda_V$ | 0.18  | unidentifiable       |
| temp        | $T$       | 0.17    | unidentifiable       |
| contact_no  | $r_{ct}$  | 0.14    | unidentifiable       |
| neighbor    | $n_{ord}$ | < 0     | unidentifiable       |

(Leave-one-out R^2 under the primary trajectory matcher; the same three
parameters clear the threshold under the end-state matcher.)

Operational thresholds: R^2 >= 0.7 identifiable, 0.3 <= R^2 < 0.7 weakly
identifiable, R^2 < 0.3 unidentifiable. The three weakly-identifiable
parameters jointly carry the wetting / dewetting structure of the CPM at
the 2D-morphology scale; the other four require additional observables
(dynamics, 3D, single-cell tracking) to be recovered.

## What this package does NOT do

- **Segmentation.** Bring your own binary masks.
- **Calibrate to absolute biological units.** Posteriors locate each
  spheroid in the closest matching synthetic regime. Roughly 90% of real
  CLL spheroids land outside the bundled library's nearest-neighbour
  envelope, so absolute values should be read as "closest synthetic
  regime", not as physical-unit point estimates. Cross-condition
  *relative* shifts on the identifiable axes are the defensible read.
- **Simulate.** The bundled CSV is the contract; the upstream CompuCell3D
  simulator is referenced via DOI in `docs/METHODS.md`.

## Documentation and provenance

- `docs/METHODS.md` - matching method, surrogate, and the upstream CompuCell3D
  simulator (DOI).
- `docs/PARAMETERS.md` - the seven CPM parameters and their symbols.
- `docs/DATA_PREPARATION.md` - accepted mask formats and folder layouts.
- `docs/CONTRIBUTING.md` - development and test instructions.
- `cll_cpm_inversion/data/synthetic_library_eda_report.md` - provenance and EDA
  of the bundled 1105-vector synthetic library (the matching contract).
- `REPO_TO_THESIS.md` - which repo paths back which thesis sections, tables, and
  figures.

## Reproducing the worked example

For the quickest end-to-end check, from a clone:

```bash
pip install -e .
python demo.py                      # one-command demo over the bundled real masks
```

or reproduce the Pass 5b J_cc table above directly from the bundled CSV:

```bash
cll-invert --features-csv examples/example_trajectories.csv \
           --trajectory --frame-col frame --out posteriors.csv
```

A second worked example, `examples/drug_panel_demo/` (four drugs, one per
mechanism class), shows the inverter running across mechanisms; see its
`README.md` for the wells and caveats.

## Citation

If you use this code, please cite:

```
D.D. Zair, "Inferring Cellular Potts Model Parameters from Chronic Lymphocytic
Leukaemia Tumour Morphology", MSc thesis, University of Amsterdam, 2026.
```

Author: D.D. Zair. Supervisor: Dr. V.S. Muniraj (University of Amsterdam).

## License

MIT. See `LICENSE`.
