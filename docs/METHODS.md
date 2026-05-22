# Methods

This package implements the inversion stage of a Cellular Potts Model
(CPM) parameter-inference pipeline for CLL spheroid morphology. The
upstream simulation and the Saltelli design that produced the bundled
synthetic library are not part of this repository; they are documented
here in enough detail that the bundled CSV can be regenerated
independently.

## 1. Inputs and outputs

Input
:   A folder of segmented spheroid masks. Each mask is a single 2D
    image (.jpg, .png, or .tif) where non-zero pixels label the
    spheroid. Two folder layouts are accepted:
    - **Flat:** one mask file per spheroid; `spheroid_id` = file stem.
    - **Nested:** one subfolder per spheroid; the subfolder contains the
      frames of the trajectory. The five features are computed per
      frame and averaged over the trajectory by default.

Output
:   A long DataFrame, one row per (spheroid, CPM parameter), with the
    posterior median, the 5th and 95th percentiles (the empirical 90%
    interval), and an identifiability flag.

## 2. Morphology features

For each mask, the largest connected component is taken to be the
spheroid and the remaining components are discarded. Five region
properties are then computed via `skimage.measure.regionprops`:

| Feature              | Definition                                       |
|---------------------|---------------------------------------------------|
| `total_area`        | Number of foreground pixels                       |
| `equivalent_diameter` | Diameter of a disk with the same area           |
| `solidity`          | Area divided by convex-hull area                  |
| `perimeter`         | Crofton perimeter estimate                        |
| `circularity`       | 4 pi A / P^2, dimensionless                       |

These five are the **operational feature set**. Eccentricity was dropped
from the operational set during RQ1 feature validation because its
ICC(3,1) across segmenters was too low to be reused safely
(0.111). Perimeter is kept because the segmenter chosen for downstream
work (U-Net 3_heavy_aug) clears the radiomics CCC bar (>= 0.85) on it.

## 3. The synthetic library

`data/synthetic_library.csv` (583 rows, 14 columns) is the
replicate-averaged final-MCS feature vector of each NaN-free Saltelli
sample, joined with the 7-parameter input vector that produced it.

| Parameter     | Symbol     | Range          | Biological meaning                |
|--------------|-----------|----------------|------------------------------------|
| `width`      | $w$       | [4, 30]        | initial spheroid radius            |
| `temp`       | $T$       | [2, 75]        | CPM thermal noise                  |
| `lambda`     | $\lambda_V$ | [0.1, 20]   | volume-conservation stiffness      |
| `contact`    | $J_{cc}$  | [1, 50]        | cell-cell adhesion energy          |
| `cm_adhesion`| $J_{cm}$  | [1, 50]        | cell-medium adhesion energy        |
| `contact_no` | $r_{ct}$  | [1, 8]         | contact-neighbour count            |
| `neighbor`   | $n_{ord}$ | [1, 7]         | neighbour-order range              |

The Saltelli design was used to compute first-order ($S_1$), total-order
($S_T$), and pairwise ($S_{ij}$) Sobol indices via the SALib estimator.
The 583 nan-free vectors come from a budget of 1152 designed vectors;
the remaining 569 are not part of the bundled library because their
simulator outputs were lost during the original sweep.

`data/sobol_indices.csv` is the per-(feature, parameter) Sobol index
table. The matcher uses the column-mean of `ST` across parameters to
weight the five features in the matching metric.

`data/identifiability_loo.csv` holds the leave-one-out R^2 and Pearson
r per parameter from the same 583-vector library, computed under the
same Sobol-weighted k-NN matcher with k = 20. These R^2 values drive
the identifiability flag.

## 4. The matcher

For each observation $x$ with feature vector
$f(x) = (f_1, ..., f_5)$, we standardise both $f(x)$ and every
synthetic feature vector $f(s)$ using the synthetic library's mean and
standard deviation, then apply the per-feature weights
$w_i = \bar{S}_T^{(i)} / \sum_j \bar{S}_T^{(j)}$ as

$$
d(x, s)^2 = \sum_{i=1}^{5} w_i \big( \tilde f_i(x) - \tilde f_i(s) \big)^2
$$

where $\tilde f$ denotes the standardised feature. The top $k$ closest
synthetic samples (default $k = 20$) form the empirical posterior
over the seven CPM parameters. The posterior median is reported as a
point estimate; the 5th and 95th percentiles form the empirical 90%
interval, which was calibrated to 89-95% coverage on synthetic LOO.

## 5. Identifiability flag

For each parameter $\theta$ we have a single LOO R^2 from the bundled
library. The operational thresholds are:

| R^2 range            | Flag                  |
|---------------------|-----------------------|
| $R^2 \geq 0.70$     | identifiable          |
| $0.30 \leq R^2 < 0.70$ | weakly identifiable |
| $R^2 < 0.30$        | unidentifiable        |

With the current bundled library, no parameter is identifiable; three
are weakly identifiable (`cm_adhesion`, `width`, `contact`), four are
unidentifiable (`lambda`, `temp`, `contact_no`, `neighbor`).

## 6. Important caveat: morphospace coverage

On the CLL spheroid datasets used in the thesis, ~90% of real
spheroids sit *above* the synthetic library's 95th-percentile LOO
nearest-neighbour distance. This means absolute parameter values
returned by this package should be read as "closest synthetic regime",
not as physical-unit point estimates. The defensible read is
cross-condition *relative* shifts on the identifiable axes.

Two natural extensions, out of scope for this repository:

1. Extend the Saltelli design's $J_{cc}$ and $J_{cm}$ ranges further
   to push the synthetic envelope into the real morphospace.
2. Render the bundled synthetic VTK trajectories as IncuCyte-style
   brightfield, segment them with the same U-Net, and re-derive the
   library to close the synthetic-to-real domain gap.

## 7. References

- SALib: Iwanaga, Usher & Herman (2022), *Journal of Open Source
  Software*, 7(76): 4584.
- Saltelli et al. (2010), *Computer Physics Communications* 181: 259-270.
- CompuCell3D: Swat et al. (2012), *Methods in Cell Biology* 110: 325-366.
