# Real spheroid mask sample

Eight U-Net-segmented CLL spheroid trajectories from patient VID1797 of
the thesis IncuCyte dataset, illustrating the package on real biological
data and the cross-condition shifts it can detect.

## Provenance

- **Source images:** IncuCyte brightfield, 1536 x 1152 px, hourly cadence,
  ~3 days per spheroid.
- **Segmenter:** U-Net 3_heavy_aug (the segmenter selected in RQ1
  feature-fidelity ranking; CCC > 0.85 against radiomics ground truth on
  the five operational features).
- **Subsampling:** every 4th hourly frame retained (~19 frames per
  trajectory) to keep the repository small. The package's default
  trajectory averaging is robust to this; absolute feature values shift
  by <2% versus the full-frame trajectory mean.

## Biological design

| Trajectories                | Stimulation     | Drug          | Dose      | Purpose                              |
|-----------------------------|----------------|---------------|----------|--------------------------------------|
| `VID1797_F1`, `VID1797_F2`  | unstimulated   | none          | -        | Unstim baseline (replicates)         |
| `VID1797_F5`, `VID1797_F6`  | IL-2/15/21/CpG | none          | -        | Stim control (perturbation effect)   |
| `VID1797_A5`                | IL-2/15/21/CpG | Entospletinib | 100 nM   | Drug rescue, highest dose            |
| `VID1797_A6`                | IL-2/15/21/CpG | Entospletinib | 10 nM    | Drug rescue, middle dose             |
| `VID1797_A7`                | IL-2/15/21/CpG | Entospletinib | 1 nM     | Drug rescue, low dose                |
| `VID1797_A8`                | IL-2/15/21/CpG | Entospletinib | 0.1 nM   | Drug rescue, expected null           |

Entospletinib is a Syk-kinase inhibitor; the IL-2/15/21/CpG cocktail
mimics tumour-microenvironment activation. In this patient the unstim
controls already sit in the dewetted regime (high $J_{cc}$); stim
*softens* cell-cell contacts and pulls $J_{cc}$ down. The biological
prediction is that Entospletinib at high dose blocks the Syk-driven
softening and rescues $J_{cc}$ back up toward (or past) the unstim
baseline, with the effect attenuating with dose.

## Important caveats

- These masks come from one patient. Cross-patient variation is large in
  CLL; do not treat the absolute parameter values as universal.
- ~90% of real CLL spheroids extrapolate the bundled synthetic library's
  nearest-neighbour envelope. Absolute parameter values should be read as
  "closest synthetic regime", not as physical-unit point estimates. See
  `docs/METHODS.md` section 6.
- The defensible read is cross-condition *relative* shifts on the
  identifiable axes (`width`, `contact`/$J_{cc}$, `cm_adhesion`/$J_{cm}$).
