# Data preparation

How to deliver segmented spheroid masks so the pipeline can process them
without surprises.

## Accepted formats

| Aspect      | Accepted                                                          |
|-------------|-------------------------------------------------------------------|
| File types  | `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff` (case-insensitive)       |
| Bit depth   | 8-bit, 16-bit, 1-bit; binary or grayscale or RGB                  |
| Pixel encoding | 0 = background, anything > 0 = spheroid                        |
| Colour      | RGB is reduced to grayscale automatically (channel mean), then thresholded > 0 |
| Multi-class label maps | Treated as a union: every non-zero class becomes foreground |

If your segmenter produces probability maps, threshold them to binary
before passing in (e.g. `prob > 0.5`).

## Folder layouts

Pick one of two:

### Layout A: flat (one mask file per spheroid)

```
my_masks/
    spheroidA.png
    spheroidB.png
    spheroidC.png
```

`spheroid_id` is the file stem (`spheroidA`, etc.). Use this when each
mask already represents a single observation; trajectory averaging is
not applied.

### Layout B: nested (one subfolder per spheroid, frames inside)

```
my_masks/
    well_F1/
        frame_00.png
        frame_01.png
        ...
    well_F2/
        frame_00.png
        ...
```

`spheroid_id` is the subfolder name. Per-frame features are averaged
into one representative vector per spheroid by default. Pass
`aggregate=False` (or `--no-aggregate` on the CLI) to invert each frame
independently.

Frame filenames just need to be sortable; the package uses lexical
order. We recommend zero-padded indices (`frame_00.png`,
`frame_01.png`, ...) or timestamp suffixes (`..._00d04h00m.png`).

## Critical rules

### Rule 1: One spheroid per image

If a mask file contains multiple connected components, the package
keeps the largest and discards the rest. This is intentional for
small specks (one or two stray pixels of post-processing noise), but
silently dropping a second genuine spheroid is dangerous. Either:

- **Crop or instance-separate** before passing in, so each file has
  exactly one spheroid; or
- Run with `--strict` (CLI) or `strict=True` (Python). Any frame with
  a second connected component of >= 5% of the largest will raise
  `ValueError` instead of being silently truncated. Use this whenever
  you expect exactly one spheroid per frame and want to be told about
  violations.

### Rule 2: Consistent resolution within a batch

Three of the five features (`total_area`, `equivalent_diameter`,
`perimeter`) are pixel-counted. The matcher z-scores observations
against the bundled synthetic library, so as long as you are
**consistent across the spheroids you want to compare**, cross-condition
*shifts* on the identifiable axes remain interpretable.

Mixing resolutions in the same batch (some 512x512, some 1536x1152)
will systematically bias the relative comparison and is not
recommended.

The bundled real-data example (`examples/real_masks/`) was segmented
at 1536x1152. If you want absolute parameter values to be loosely
comparable to numbers reported in the thesis, send masks at a similar
resolution. **Absolute values are anyway only "closest synthetic
regime" because ~90% of real CLL spheroids extrapolate the bundled
library's nearest-neighbour envelope** (see `METHODS.md` section 6),
so this is a soft preference, not a hard requirement.

### Rule 3: Whole spheroids only

Crop windows should contain the whole spheroid plus background, not a
clipped piece. A spheroid touching the image edge will have a
truncated perimeter and underestimated solidity. The package does not
auto-detect or auto-correct edge clipping; you must crop with a
reasonable margin upstream.

## Quick sanity checks

Before sending a batch, run:

```python
from cll_cpm_inversion import features_from_folder
df = features_from_folder("my_masks/", strict=True)
print(df.describe())
```

What to look for:
- `strict=True` did not raise -> no multi-component frames
- `total_area` distribution looks plausible (no rows near zero unless
  you expect dead spheroids)
- `solidity` between ~0.7 and ~1.0 for healthy spheroids
- `circularity` between ~0.2 (very dewetted) and ~0.9 (very round)

## What NOT to send

- Probability maps (threshold first)
- Multi-spheroid images without cropping (use `--strict` to catch these)
- Inverted masks (background = 255, foreground = 0). The package treats
  zero as background; if you have inverted masks, invert before saving.
- Mixed resolutions in the same batch (compare apples to apples).
- Edge-clipped spheroids (crop with a margin first).

## Minimal example payload

If you send the package owner / collaborator a zip:

```
my_batch.zip
    metadata.csv            # spheroid_id, condition, patient, ...
    masks/
        sample_001/         # one trajectory
            00.png
            01.png
            02.png
            ...
        sample_002/
            00.png
            ...
        ...
```

`metadata.csv` should contain a `spheroid_id` column that matches the
subfolder names so the recipient can join condition labels to the
posterior output without guessing.
