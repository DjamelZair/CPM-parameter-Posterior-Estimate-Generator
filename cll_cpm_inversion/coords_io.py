"""Coordinate-CSV input: invert a ``(cell_id, x, y)`` point list.

A coordinate CSV is the portable input format the pipeline accepts alongside
mask images. Each row is one foreground pixel of a spheroid::

    cell_id,x,y
    W001,512,488
    W001,513,488
    ...

``cell_id`` (accepted aliases: ``spheroid_id``, ``id``, ``label``) is the
spheroid identifier; every row sharing an id belongs to one spheroid. ``x`` and
``y`` (aliases ``col``/``cx`` and ``row``/``cy``) are the pixel column and row.
Each spheroid's points are rasterised into a binary mask and run through the
same :func:`~cll_cpm_inversion.features.extract_features_from_mask` as the image
path, so a coordinate CSV feeds straight into the inversion. The reported
features are translation invariant, so absolute coordinate offsets do not matter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .features import OPERATIONAL_FEATURES, extract_features_from_mask

ID_ALIASES = ("cell_id", "cellid", "spheroid_id", "spheroidid", "id", "label")
X_ALIASES = ("x", "col", "cx", "px")
Y_ALIASES = ("y", "row", "cy", "py")

CoordsSource = Union[str, Path, pd.DataFrame]


def read_coords(src: CoordsSource, sep: str | None = None) -> pd.DataFrame:
    """Read a coordinate CSV and normalise its columns to ``spheroid_id, x, y``.

    ``sep=None`` auto-detects the delimiter (``,`` / ``;`` / tab).
    """
    if isinstance(src, pd.DataFrame):
        df = src.copy()
    else:
        df = pd.read_csv(src, sep=sep, engine="python")
    lut = {str(c).lower().strip(): c for c in df.columns}

    def pick(aliases: tuple[str, ...], name: str) -> str:
        for a in aliases:
            if a in lut:
                return lut[a]
        raise KeyError(
            f"coordinate CSV needs a '{name}' column; got {list(df.columns)} "
            f"(accepted aliases: {aliases})"
        )

    out = df.rename(columns={pick(ID_ALIASES, "id"): "spheroid_id",
                             pick(X_ALIASES, "x"): "x",
                             pick(Y_ALIASES, "y"): "y"})
    return out[["spheroid_id", "x", "y"]]


def mask_from_points(x, y, margin: int = 2) -> np.ndarray:
    """Rasterise one spheroid's foreground points into a binary mask.

    Points are shifted to their own bounding box (plus ``margin`` px of
    padding) so the canvas stays small; the computed features are translation
    invariant, so this does not change them.
    """
    x = np.rint(np.asarray(x, dtype=float)).astype(int)
    y = np.rint(np.asarray(y, dtype=float)).astype(int)
    x = x - x.min() + margin
    y = y - y.min() + margin
    H = int(y.max()) + 1 + margin
    W = int(x.max()) + 1 + margin
    m = np.zeros((H, W), dtype=np.uint8)
    m[y, x] = 1
    return m


def features_from_coords(src: CoordsSource, strict: bool = False,
                         sep: str | None = None) -> pd.DataFrame:
    """``(cell_id, x, y)`` CSV -> one feature row per spheroid id.

    Returns a DataFrame with ``spheroid_id`` and the five
    ``OPERATIONAL_FEATURES`` columns, ready for ``infer_from_features``.
    """
    df = read_coords(src, sep=sep)
    rows = []
    for sid, g in df.groupby("spheroid_id", sort=False):
        mask = mask_from_points(g["x"].to_numpy(), g["y"].to_numpy())
        feats = extract_features_from_mask(mask, strict=strict)
        feats["spheroid_id"] = sid
        rows.append(feats)
    if not rows:
        raise ValueError("coordinate CSV produced no spheroids (empty input)")
    return pd.DataFrame(rows)[["spheroid_id"] + OPERATIONAL_FEATURES]
