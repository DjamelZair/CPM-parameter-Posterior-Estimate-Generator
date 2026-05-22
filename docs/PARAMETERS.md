# CPM parameter reference

The seven parameters returned by this package map onto a standard CPM
energy functional

$$
H = \sum_{(i,j) \in \text{neighbours}} J\big(\tau(\sigma_i), \tau(\sigma_j)\big)
\big(1 - \delta_{\sigma_i,\sigma_j}\big)
+ \sum_{\sigma > 0} \lambda_V \big( V(\sigma) - V_T \big)^2
$$

with the Metropolis acceptance probability tempered by $T$.

## Parameter table

| Code           | Symbol     | Range in library | Role                                                                                       |
|---------------|-----------|------------------|---------------------------------------------------------------------------------------------|
| `width`       | $w$       | [4, 30]          | Initial spheroid radius. Drives size-family features.                                       |
| `temp`        | $T$       | [2, 75]          | Boltzmann temperature in the Metropolis rule. Lower $T$ = stiffer dynamics.                 |
| `lambda`      | $\lambda_V$ | [0.1, 20]      | Volume-conservation stiffness. Penalises deviations from target volume $V_T$.               |
| `contact`     | $J_{cc}$  | [1, 50]          | Cell-cell adhesion energy. Higher $J_{cc}$ = stronger preference *against* cell-cell contacts (the model penalises mismatched neighbours). |
| `cm_adhesion` | $J_{cm}$  | [1, 50]          | Cell-medium adhesion energy. Together with $J_{cc}$ it sets the wetting / dewetting axis.   |
| `contact_no`  | $r_{ct}$  | [1, 8]           | Contact-neighbour count threshold.                                                          |
| `neighbor`    | $n_{ord}$ | [1, 7]           | Neighbour-order range (radius of the neighbour shell).                                      |

## Reading posteriors

The matcher returns one empirical posterior per spheroid. For each
parameter, the package reports:

| Column            | Meaning                                                              |
|-------------------|----------------------------------------------------------------------|
| `median`          | Posterior median (point estimate).                                   |
| `q05`, `q95`      | Empirical 5th and 95th percentile (90% interval).                    |
| `q25`, `q75`      | Empirical IQR.                                                       |
| `n_matches`       | Number of top-k matches contributing to this row (default 20).       |
| `loo_r2`          | Library leave-one-out R^2 for this parameter.                        |
| `identifiability` | `identifiable` / `weakly identifiable` / `unidentifiable`.           |

Rows tagged `unidentifiable` should not be interpreted as point
estimates; their values are essentially the parameter's prior
expectation. They are still included in the output so that user code
can verify the flag.

## Interpretation rules of thumb

- **Width and adhesions jointly explain size and shape.** Width owns the
  size family (total_area, equivalent_diameter); $J_{cm}$ owns the shape
  family (solidity, circularity, perimeter); $J_{cc}$ enters as a
  pairwise interaction with $J_{cm}$ around the wetting / dewetting
  transition. A spheroid with high $J_{cc}$ and low $J_{cm}$ sits in the
  dewetted regime: low circularity, high perimeter, ragged outline.
- **$\lambda_V$, $T$, $r_{ct}$, $n_{ord}$ are unrecoverable from a single
  2D snapshot.** Their posteriors will spread across the whole prior
  range; do not read them as biological readouts.
- **Cross-condition shifts are more defensible than absolute values.**
  Comparing two posteriors (e.g. stimulated vs control) on the
  identifiable axes is the recommended use; reporting one absolute value
  in isolation is not.
