#!/usr/bin/env python
"""Diagnostic grid: PoE-compose vs generator-overlay vs real both-disease.

Question the grid answers at a glance: does COMPOSE render ONE coherent body
carrying BOTH findings (enlarged heart + basal effusion), versus the OVERLAY's
two-body ghosting?

3 rows x 5 cols, native resolution, grayscale, nearest-neighbour (no smoothing).
Row 1: PoE-compose (outputs/compose/compose/w1p0)
Row 2: generator-overlay (outputs/overlay_gen/overlay)
Row 3: real both-disease (data/nih/images_both)

Re-run: micromamba run -n jaxstack python scripts/diag_compose_vs_overlay.py
"""
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 5
ROWS = [
    ("PoE-compose\n(w1p0)",        "outputs/compose/compose/w1p0/*.png"),
    ("Generator-overlay",          "outputs/overlay_gen/overlay/*.png"),
    ("Real both-disease\n(NIH)",   "data/nih/images_both/*.png"),
]
OUT = "figures/diag_compose_vs_overlay.png"

fig, axes = plt.subplots(len(ROWS), N, figsize=(N * 2.6, len(ROWS) * 2.8))

for r, (label, pat) in enumerate(ROWS):
    files = sorted(glob.glob(pat))[:N]
    for c in range(N):
        ax = axes[r, c]
        ax.set_xticks([]); ax.set_yticks([])
        if c < len(files):
            img = np.asarray(plt.imread(files[c]))
            if img.ndim == 3:
                img = img[..., 0]
            ax.imshow(img, cmap="gray", interpolation="nearest", aspect="equal")
            ax.set_title(os.path.basename(files[c]), fontsize=6.5, pad=2)
        if c == 0:
            ax.set_ylabel(label, fontsize=10, fontweight="bold", rotation=90,
                          labelpad=8, va="center")

fig.suptitle(
    "ONE coherent body with BOTH findings (compose) vs two-body ghosting (overlay)?\n"
    "Look for: single chest carrying enlarged heart + basal effusion, "
    "vs doubled/ghosted silhouettes",
    fontsize=11, y=0.99,
)
fig.tight_layout(rect=[0.02, 0.0, 1.0, 0.94])
os.makedirs("figures", exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}  ({os.path.getsize(OUT)/1e3:.0f} KB)")
