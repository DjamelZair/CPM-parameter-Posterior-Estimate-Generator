"""`cll-invert` command-line entry point.

Usage:
    cll-invert MASKS_DIR [--out OUT.csv] [-k 20]
                         [--no-aggregate]
                         [--features-csv FILE]

Either MASKS_DIR or --features-csv must be provided.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .pipeline import infer_from_features, infer_from_masks


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cll-invert",
        description="Invert spheroid morphology into CPM parameter posteriors.",
    )
    p.add_argument("masks_dir", nargs="?", default=None,
                   help="Folder of masks (.jpg/.png/.tif). "
                        "Flat or one subfolder per spheroid.")
    p.add_argument("--features-csv", default=None,
                   help="Skip feature extraction; read a pre-built feature "
                        "CSV (must contain spheroid_id and the five "
                        "OPERATIONAL_FEATURES columns).")
    p.add_argument("--coords-csv", default=None,
                   help="Invert a (cell_id, x, y) coordinate CSV: each distinct "
                        "id is one spheroid and its rows are the foreground "
                        "pixels (aliases: spheroid_id/id/label, col/cx, row/cy).")
    p.add_argument("--trajectory", action="store_true",
                   help="Primary matcher: match the whole tau-registered "
                        "trajectory against the bundled library trajectories "
                        "instead of the end-state. Use with a Layout-B masks "
                        "folder (one subfolder per spheroid) or a feature CSV "
                        "with multiple frames per spheroid_id (see --frame-col).")
    p.add_argument("--frame-col", default=None,
                   help="With --trajectory + --features-csv, the column giving "
                        "frame order within each spheroid.")
    p.add_argument("--out", default="posteriors.csv",
                   help="Output CSV path (default: posteriors.csv)")
    p.add_argument("-k", "--k", type=int, default=20,
                   help="Top-k nearest synthetic matches (default: 20)")
    p.add_argument("--no-aggregate", action="store_true",
                   help="With per-spheroid trajectory folders, invert each "
                        "frame independently instead of averaging.")
    p.add_argument("--strict", action="store_true",
                   help="Error out (instead of silently keeping the largest) "
                        "if any frame contains more than one large "
                        "connected component (>= 5%% of the largest). Use "
                        "this when you expect exactly one spheroid per "
                        "frame and want to be told about violations.")
    p.add_argument("--save-features", default=None,
                   help="Also write the extracted feature table to this CSV.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.masks_dir and not args.features_csv and not args.coords_csv:
        print("ERROR: must supply masks_dir, --features-csv, or --coords-csv",
              file=sys.stderr)
        return 2

    if args.trajectory:
        from .trajectory import infer_from_trajectory, infer_from_mask_trajectories
        if args.features_csv:
            frames = pd.read_csv(args.features_csv)
            print(f"Trajectory matching {frames[ 'spheroid_id'].nunique()} "
                  f"spheroids from {args.features_csv}")
            summary = infer_from_trajectory(frames, frame_col=args.frame_col, k=args.k)
        elif args.masks_dir:
            print(f"Trajectory matching from {args.masks_dir} (Layout B)...")
            summary = infer_from_mask_trajectories(args.masks_dir, k=args.k,
                                                   strict=args.strict)
        else:
            print("ERROR: --trajectory needs a masks folder or --features-csv",
                  file=sys.stderr)
            return 2
    elif args.coords_csv:
        from .coords_io import features_from_coords
        features = features_from_coords(args.coords_csv, strict=args.strict)
        print(f"Built features for {len(features)} spheroids from "
              f"{args.coords_csv}")
        summary = infer_from_features(features, k=args.k)
    elif args.features_csv:
        features = pd.read_csv(args.features_csv)
        print(f"Loaded {len(features)} feature rows from {args.features_csv}")
        summary = infer_from_features(features, k=args.k)
    else:
        masks_dir = Path(args.masks_dir)
        if not masks_dir.is_dir():
            print(f"ERROR: not a directory: {masks_dir}", file=sys.stderr)
            return 2
        print(f"Extracting features from {masks_dir}"
              f"{' (strict mode)' if args.strict else ''}...")
        try:
            summary, features = infer_from_masks(
                masks_dir, k=args.k,
                aggregate_trajectories=not args.no_aggregate,
                strict=args.strict,
                return_features=True,
            )
        except ValueError as exc:
            print(f"ERROR (strict mode): {exc}", file=sys.stderr)
            return 3
        if args.save_features:
            Path(args.save_features).parent.mkdir(parents=True, exist_ok=True)
            features.to_csv(args.save_features, index=False)
            print(f"Saved features -> {args.save_features}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"Saved posterior summary -> {out_path}  ({len(summary)} rows)")

    ident = summary[summary["identifiability"] != "unidentifiable"]
    print(f"\nIdentifiable + weakly-identifiable posterior medians "
          f"({ident['parameter'].nunique()} parameters, "
          f"{summary['spheroid_id'].nunique()} spheroids):")
    pivot = (ident.pivot(index="spheroid_id",
                         columns="parameter", values="median")
                  .round(2))
    print(pivot.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
