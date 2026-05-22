"""Extract the five operational morphology features from a binary mask.

Supported inputs:
    - single 2D mask file (.jpg, .jpeg, .png, .tif, .tiff)
    - folder of mask files (flat = one spheroid per file, or one subfolder
      per spheroid trajectory with frames inside)
    - in-memory 2D numpy array

Foreground convention: any non-zero pixel is treated as spheroid. RGB
images are first converted to grayscale and thresholded at >0. If a frame
contains multiple disconnected components, the largest connected region
is used as the spheroid (small specks are discarded).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from skimage import measure
from skimage.morphology import label as cc_label

OPERATIONAL_FEATURES = [
    "total_area",
    "equivalent_diameter",
    "solidity",
    "perimeter",
    "circularity",
]

IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def _binarise(arr: np.ndarray) -> np.ndarray:
    """Reduce to a clean boolean 2D mask: keep the largest CC, drop the rest."""
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    mask = arr > 0
    if not mask.any():
        return mask
    labelled = cc_label(mask, connectivity=2)
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0
    keep = int(sizes.argmax())
    return labelled == keep


def _load_image(path: Path) -> np.ndarray:
    """Load .jpg / .png / .tif / .tiff as a 2D numpy array."""
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        try:
            import tifffile
            arr = tifffile.imread(str(path))
        except Exception:
            arr = np.array(Image.open(path))
    else:
        arr = np.array(Image.open(path))
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def extract_features_from_mask(mask) -> dict:
    """Compute the five operational features from one mask.

    Parameters
    ----------
    mask : numpy.ndarray | str | pathlib.Path
        Either a 2D mask array or a path to an image file.

    Returns
    -------
    dict with keys: total_area, equivalent_diameter, solidity, perimeter,
    circularity. NaN-filled when the mask is empty.
    """
    if isinstance(mask, (str, Path)):
        mask = _load_image(Path(mask))
    elif not isinstance(mask, np.ndarray):
        raise TypeError(f"mask must be ndarray or path, got {type(mask)}")

    binary = _binarise(mask)
    if not binary.any():
        return {f: float("nan") for f in OPERATIONAL_FEATURES}

    props = measure.regionprops(binary.astype(np.uint8))[0]
    area = float(props.area)
    perim = float(props.perimeter) if props.perimeter > 0 else float("nan")
    eq_d = float(props.equivalent_diameter)
    solidity = float(props.solidity)
    if perim > 0:
        circularity = float(4.0 * np.pi * area / (perim * perim))
    else:
        circularity = float("nan")
    return {
        "total_area":          area,
        "equivalent_diameter": eq_d,
        "solidity":            solidity,
        "perimeter":           perim,
        "circularity":         circularity,
    }


def _list_image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_SUFFIXES)


def _aggregate_trajectory(rows: list[dict]) -> dict:
    """Average per-frame features over a trajectory.

    The synthetic library is built from the final-MCS replicate-averaged
    feature vector per Saltelli sample, so the natural matching input is
    one representative vector per spheroid. We take the trajectory mean
    (NaN-safe) here; users who want frame-resolved inversion should call
    `infer_from_features` on the un-aggregated frame table directly.
    """
    df = pd.DataFrame(rows)
    return {f: float(df[f].mean(skipna=True)) for f in OPERATIONAL_FEATURES}


def features_from_folder(folder, aggregate: bool = True) -> pd.DataFrame:
    """Walk a folder and return one feature row per spheroid.

    Layout A (flat):  one image file per spheroid -> spheroid_id = filename stem
    Layout B (nested): one subfolder per spheroid -> spheroid_id = subfolder name,
                       each subfolder holds the frames of that trajectory

    Parameters
    ----------
    folder : str | Path
        Root folder.
    aggregate : bool, default True
        If True and layout B, average the per-frame features into a single
        representative vector per spheroid. If False, return one row per
        (spheroid_id, frame).

    Returns
    -------
    pandas.DataFrame
        Columns: spheroid_id [, frame] + OPERATIONAL_FEATURES.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    subdirs = [d for d in folder.iterdir() if d.is_dir()]
    rows: list[dict] = []

    if subdirs:
        for sd in sorted(subdirs):
            frames = _list_image_files(sd)
            if not frames:
                continue
            traj_rows = []
            for i, fpath in enumerate(frames):
                feats = extract_features_from_mask(fpath)
                feats.update({"spheroid_id": sd.name, "frame": i,
                              "source_file": str(fpath)})
                traj_rows.append(feats)
            if aggregate:
                agg = _aggregate_trajectory(traj_rows)
                agg["spheroid_id"] = sd.name
                agg["n_frames"] = len(traj_rows)
                rows.append(agg)
            else:
                rows.extend(traj_rows)
    else:
        files = _list_image_files(folder)
        if not files:
            raise FileNotFoundError(
                f"No image files (.jpg/.png/.tif/.tiff) found in {folder}")
        for fpath in files:
            feats = extract_features_from_mask(fpath)
            feats.update({"spheroid_id": fpath.stem,
                          "source_file": str(fpath)})
            rows.append(feats)

    df = pd.DataFrame(rows)
    leading = ["spheroid_id"] + (["frame"] if "frame" in df.columns else [])
    return df[leading + OPERATIONAL_FEATURES +
              [c for c in df.columns
               if c not in leading + OPERATIONAL_FEATURES]]
