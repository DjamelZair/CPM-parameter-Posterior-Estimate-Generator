# Data provenance and canonical-value note

This repository bundles the synthetic library and its derived tables so the
inversion tool runs offline. One file pair needs an explicit canonical note.

## `identifiability_loo.csv` (canonical) vs `identifiability_loo_legacy.csv`

`identifiability_loo.csv` holds the leave-one-out R^2 / Pearson r per parameter
that the bundled tool emits as `loo_r2` and uses to set the identifiability
flag. It now carries the **canonical tau-primary** set, matching the thesis
(`table5_identifiability_tau.csv`, `weighting = w_mean`):

| param | symbol | canonical R^2 | flag |
|---|---|---|---|
| width | w | 0.635 | weakly identifiable |
| cm_adhesion | J_cm | 0.615 | weakly identifiable |
| contact | J_cc | 0.344 | weakly identifiable |
| contact_no | r_ct | 0.167 | unidentifiable |
| temp | T | 0.164 | unidentifiable |
| lambda | lambda_V | 0.126 | unidentifiable |
| neighbor | n_ord | -0.069 | unidentifiable |

`identifiability_loo_legacy.csv` is the **earlier, non-canonical** leave-one-out
computation (e.g. w 0.544, J_cc 0.384), preserved unchanged for transparency and
**not deleted**. The two sets come from the same tau-trajectory matcher on the
same 1105-vector Saltelli library and differ only in magnitude, not in the
qualitative result: the identifiability flags are **identical** under either set
(`cm_adhesion`, `width`, `contact` weakly identifiable for 0.30 <= R^2 < 0.70;
`lambda`, `temp`, `contact_no`, `neighbor` unidentifiable). Cite the canonical
set above for thesis R^2 values; the legacy file is kept only for reproducibility
of the earlier numbers.

No per-patient, per-well, or otherwise patient-identifying data is included in
this repository.
