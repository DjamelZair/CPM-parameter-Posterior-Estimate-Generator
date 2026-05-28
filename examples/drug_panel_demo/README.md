# Multi-drug demo (four mechanism classes)

Four U-Net-segmented CLL spheroid trajectories, one per mechanism class, to show
the inverter running across drugs (not just the Entospletinib dose-response in
`../real_masks/`). Each subfolder is one spheroid (Layout-B: one mask PNG per
frame, time-encoded names that sort chronologically).

| spheroid | drug | mechanism | dose (nM) | patient | frames |
|---|---|---|---|---|---|
| VID1087_A2  | ibrutinib   | BTK inhibitor (BCR axis)        | 1000 | 2089   | 23 |
| VID1087_A5  | idelalisib  | PI3K-delta inhibitor (BCR axis) | 100  | 2089   | 23 |
| VID1087_A11 | Bay 11-7082 | NF-kB inhibitor                 | 1000 | 2089   | 23 |
| VID1873_D4  | IT1T        | CXCR4 antagonist (microenv.)    | 10   | 706_t1 | 112 |

## Run

```bash
# trajectory matcher (primary), one posterior per spheroid:
cll-invert examples/drug_panel_demo/ --trajectory --out drug_panel_demo_posteriors.csv
```

```python
import cll_cpm_inversion as ci
post = ci.infer_from_mask_trajectories("examples/drug_panel_demo/", k=20)
print(post[post.identifiability != "unidentifiable"]
      .pivot(index="spheroid_id", columns="parameter", values="median"))
```

## Important caveats (state these when presenting)

- These wells are **unstimulated, single-dose** measurements. Unlike the
  Entospletinib block in `../real_masks/`, there is no stimulated baseline here,
  so this is a "the tool runs across mechanisms" demo, **not** a drug-rescue
  result.
- Three wells are patient 2089; IT1T is patient 706_t1. Patient-to-patient
  morphological variation is large (often larger than the drug effect), so do
  **not** read these four as a like-for-like cross-drug ranking. Compare within a
  patient, and read only the identifiable axes (J_cm, width, J_cc).
