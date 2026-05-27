#!/usr/bin/env python3
"""One-command demo for cll-cpm-inversion.

    python demo.py

Runs the primary (tau-registered) trajectory matcher on the bundled
real CLL-spheroid mask trajectories in ``examples/real_masks/``, prints the
inferred CPM-parameter posteriors with identifiability flags, writes
``demo_posteriors.csv``, and -- if matplotlib is installed -- saves
``demo_posteriors.png``. No network, no external data; runs in a few seconds.
"""
from __future__ import annotations

from pathlib import Path

import cll_cpm_inversion as ci

HERE = Path(__file__).resolve().parent
MASKS = HERE / "examples" / "real_masks"
IDENT = ["cm_adhesion", "width", "contact"]          # the recoverable parameters
LABEL = {"cm_adhesion": "J_cm (cell-medium adh.)",
         "width": "w (cell width)", "contact": "J_cc (cell-cell adh.)"}


def main() -> int:
    print("=" * 66)
    print(" cll-cpm-inversion  --  demo")
    print("=" * 66)
    print(f"input : {MASKS}")
    print("        bundled U-Net-segmented CLL spheroid trajectories")
    print("method: tau-registered trajectory matching vs the 1105-sample library\n")

    post = ci.infer_from_mask_trajectories(str(MASKS), k=20)
    n = post["spheroid_id"].nunique()
    print(f"inverted {n} spheroid trajectories.\n")

    show = post[post.parameter.isin(IDENT)]
    print("Inferred identifiable CPM parameters  (median [5th-95th percentile]):")
    print("-" * 66)
    for sid, g in show.groupby("spheroid_id"):
        parts = []
        for p in IDENT:
            r = g[g.parameter == p].iloc[0]
            parts.append(f"{p}={r['median']:.1f} [{r['q05']:.0f},{r['q95']:.0f}]")
        print(f"  {sid:13s} " + "   ".join(parts))
    print("-" * 66)
    unident = sorted(post[post.identifiability == "unidentifiable"]["parameter"].unique())
    print(f"flagged unidentifiable (not interpreted): {', '.join(unident)}\n")

    out_csv = HERE / "demo_posteriors.csv"
    post.to_csv(out_csv, index=False)
    print(f"full posteriors written to  {out_csv.name}")

    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
        for ax, p in zip(axes, IDENT):
            g = show[show.parameter == p].sort_values("median")
            y = np.arange(len(g))
            ax.errorbar(g["median"], y,
                        xerr=[g["median"] - g["q05"], g["q95"] - g["median"]],
                        fmt="o", color="#004444", ecolor="#88a", capsize=2)
            ax.set_yticks(y); ax.set_yticklabels(g["spheroid_id"], fontsize=8)
            ax.set_title(LABEL[p], fontsize=10)
            ax.set_xlabel("inferred value")
            ax.grid(axis="x", alpha=0.3)
        fig.suptitle("Inferred identifiable CPM parameters per spheroid "
                     "(median, 90% interval)", fontsize=11)
        out_png = HERE / "demo_posteriors.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"figure written to           {out_png.name}")
    except ImportError:
        print("(install matplotlib for the figure:  pip install matplotlib)")
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
