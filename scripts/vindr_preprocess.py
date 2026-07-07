"""Preprocess VinDr-512 images with the IDENTICAL transform used for NIH CXR.

Transform (mirrors precompute_latents.py::_load_image):
  1. Image.open(path).convert("L")          # grayscale, mode L
  2. Polarity check — invert if MONOCHROME1 (lung fields brighter than heart)
  3. img.resize((512, 512), Image.LANCZOS)  # only if not already 512²
  4. pixel/127.5 - 1.0                      # → float32 ∈ [-1, 1]

MONOCHROME1 detection
---------------------
CXR scanners that store DICOM with PhotometricInterpretation=MONOCHROME1 produce
PNGs where lung fields are BRIGHT and the skeleton/heart is DARK — the opposite of
the NIH convention used by all downstream instruments (_peri_cardiac_opacity,
opacity_score.py, PSPNet CTR).  Detection uses an anatomical signal:

    lung_mean  = mean pixel in upper-lateral strips (avoids mediastinum)
    heart_mean = mean pixel in centre column (heart + mediastinum)

    MONO2 (NIH): dark lungs < bright heart  →  lung_mean < heart_mean
    MONO1 (inv): bright lungs > dark heart  →  lung_mean > heart_mean + MARGIN

MARGIN (default 40) is calibrated on the full VinDr-512 train split where genuine
inversions were absent (max observed margin 26.3, caused by pathological haziness,
not polarity).  True MONO1 images exhibit margins of 60-150.

Outputs to --out-dir (default data/vindr512/std/train/):
  <image_id>.png  — uint8 grayscale 512×512 (lossless; normalization is
                    at-encode-time, identical to NIH pipeline)

Label assignment from train.csv (one-hot priority, mirroring NIH):
  • "Cardiomegaly"    → cardio   (skip if ALSO has Pleural effusion)
  • "Pleural effusion" → effusion (skip if ALSO has Cardiomegaly)
  • ALL annotations "No finding" → normal
  • Anything else    → skipped (other pathologies)

A manifest CSV is written to out-dir/../manifest.csv with columns:
  image_id, label, split, src_path, out_path, polarity_flipped

Usage:
    python scripts/vindr_preprocess.py \\
        --vindr-dir data/vindr512/vinbigdata \\
        --out-dir   data/vindr512/std \\
        --nih-dir   data/nih/images \\
        --plot-dir  results/vindr_transfer/histograms
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------

CARDIO_TAG   = "Cardiomegaly"
EFFUSION_TAG = "Pleural effusion"
NORMAL_TAG   = "No finding"

LABEL_MAP = {"normal": 0, "cardio": 1, "effusion": 2}


def _assign_labels(csv_path: Path) -> dict[str, str]:
    """Return {image_id: label_str} for images that map cleanly to one class."""
    per_image: dict[str, set[str]] = defaultdict(set)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            img_id = row["image_id"].strip()
            per_image[img_id].add(row["class_name"].strip())

    result: dict[str, str] = {}
    for img_id, labels in per_image.items():
        has_cardio    = CARDIO_TAG   in labels
        has_effusion  = EFFUSION_TAG in labels
        if has_cardio and has_effusion:   # co-morbid — exclude
            continue
        if has_cardio:
            result[img_id] = "cardio"
        elif has_effusion:
            result[img_id] = "effusion"
        elif labels == {NORMAL_TAG}:
            result[img_id] = "normal"
    return result


# ---------------------------------------------------------------------------
# Polarity detection + correction
# ---------------------------------------------------------------------------

# Anatomical margin threshold (lung_mean - heart_mean) that triggers inversion.
# True MONO1 images: margin 60-150.  Pathological noise in MONO2: max ~26.
MONO1_MARGIN = 40


def _polarity_margin(arr: np.ndarray) -> float:
    """Return lung_mean - heart_mean for a uint8 (H, W) grayscale array.

    Positive → lung brighter than heart (MONO1 indicator).
    Negative → lung darker than heart (MONO2 / NIH convention).
    """
    h, w = arr.shape
    left_lung  = arr[int(h * 0.10): int(h * 0.50), int(w * 0.04): int(w * 0.25)]
    right_lung = arr[int(h * 0.10): int(h * 0.50), int(w * 0.75): int(w * 0.96)]
    lung_mean  = (left_lung.mean() + right_lung.mean()) / 2.0
    heart      = arr[int(h * 0.25): int(h * 0.75), int(w * 0.35): int(w * 0.65)]
    return float(lung_mean - heart.mean())


def _needs_inversion(arr: np.ndarray, margin_thresh: float = MONO1_MARGIN) -> bool:
    return _polarity_margin(arr) > margin_thresh


# ---------------------------------------------------------------------------
# Image transform (identical to precompute_latents._load_image)
# ---------------------------------------------------------------------------

RES = 512


def _load_normalized(path: Path, margin_thresh: float = MONO1_MARGIN) -> tuple[np.ndarray, bool]:
    """Return (float32 (512,512) in [-1,1], flipped:bool).

    Applies NIH-identical transform with MONO1 correction before normalization.
    """
    img = Image.open(path).convert("L")
    if img.width != RES or img.height != RES:
        img = img.resize((RES, RES), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    flipped = _needs_inversion(arr.astype(np.uint8), margin_thresh)
    if flipped:
        arr = 255.0 - arr
    return arr / 127.5 - 1.0, flipped


def _save_std(arr_norm: np.ndarray, out_path: Path) -> None:
    """Save float32 [-1,1] array as uint8 PNG (reverses normalization)."""
    uint8 = np.clip((arr_norm + 1.0) * 127.5, 0, 255).astype(np.uint8)
    Image.fromarray(uint8, mode="L").save(out_path, compress_level=1)


# ---------------------------------------------------------------------------
# Before/after panel for flipped films
# ---------------------------------------------------------------------------

def _plot_flip_panel(
    flipped_paths: list[Path],
    plot_dir: Path,
    margin_thresh: float,
    n_show: int = 6,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)

    show = flipped_paths[:n_show]
    ncols = len(show)
    if ncols == 0:
        return

    fig, axes = plt.subplots(3, ncols, figsize=(3.5 * ncols, 10))
    if ncols == 1:
        axes = [[axes[0]], [axes[1]], [axes[2]]]

    fig.suptitle(
        f"MONOCHROME1 correction  (lung − heart margin > {margin_thresh})\n"
        "Row 1: BEFORE (raw)   Row 2: AFTER (255 − pixel)   Row 3: histogram overlay",
        fontsize=10,
    )

    for col, path in enumerate(show):
        img = Image.open(path).convert("L")
        if img.width != RES or img.height != RES:
            img = img.resize((RES, RES), Image.LANCZOS)
        before = np.array(img, dtype=np.float32)
        after  = 255.0 - before
        margin = _polarity_margin(before.astype(np.uint8))

        ax0 = axes[0][col]
        ax0.imshow(before, cmap="gray", vmin=0, vmax=255)
        ax0.set_title(f"BEFORE\n{path.name[:18]}\nmargin={margin:.1f}", fontsize=7)
        ax0.axis("off")

        ax1 = axes[1][col]
        ax1.imshow(after, cmap="gray", vmin=0, vmax=255)
        ax1.set_title("AFTER (inverted)", fontsize=7)
        ax1.axis("off")

        ax2 = axes[2][col]
        ax2.hist(before.ravel(), bins=64, range=(0, 255),
                 alpha=0.5, color="#5577cc", label="before", edgecolor="none")
        ax2.hist(after.ravel(),  bins=64, range=(0, 255),
                 alpha=0.5, color="#cc5533", label="after",  edgecolor="none")
        ax2.legend(fontsize=6)
        ax2.set_xlabel("pixel (uint8)", fontsize=7)
        ax2.tick_params(labelsize=6)

    plt.tight_layout()
    out = plot_dir / "vindr_polarity_flip_panel.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] flip panel → {out}")


# ---------------------------------------------------------------------------
# Histogram plot
# ---------------------------------------------------------------------------

def _plot_histograms(
    vindr_paths: list[Path],
    nih_paths: list[Path],
    plot_dir: Path,
    n: int = 3,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    v_sample  = rng.sample(vindr_paths, min(n, len(vindr_paths)))
    nih_sample = rng.sample(nih_paths,  min(n, len(nih_paths)))

    fig, axes = plt.subplots(n, 3, figsize=(13, 4 * n))
    fig.suptitle(
        "Pixel intensity after NIH transform (÷127.5 − 1)\n"
        "VinDr before (uint8 0–255) | VinDr after (float32 −1–1) | NIH after (float32 −1–1)",
        fontsize=11,
    )

    bins = 128
    for i in range(n):
        v_raw = np.array(Image.open(v_sample[i]).convert("L"), dtype=np.float32)
        ax = axes[i, 0]
        ax.hist(v_raw.ravel(), bins=bins, range=(0, 255), color="#5599cc", edgecolor="none")
        ax.set_title(f"VinDr RAW  — {v_sample[i].name[:20]}", fontsize=8)
        ax.set_xlabel("pixel (uint8 0–255)")
        ax.set_ylabel("count")

        v_norm, _ = _load_normalized(v_sample[i])
        ax = axes[i, 1]
        ax.hist(v_norm.ravel(), bins=bins, range=(-1, 1), color="#55aa55", edgecolor="none")
        ax.set_title(f"VinDr NORM — {v_sample[i].name[:20]}", fontsize=8)
        ax.set_xlabel("pixel (float32 −1–1)")

        nih_norm, _ = _load_normalized(nih_sample[i])
        ax = axes[i, 2]
        ax.hist(nih_norm.ravel(), bins=bins, range=(-1, 1), color="#cc6655", edgecolor="none")
        ax.set_title(f"NIH NORM   — {nih_sample[i].name[:20]}", fontsize=8)
        ax.set_xlabel("pixel (float32 −1–1)")

    plt.tight_layout()
    out = plot_dir / "vindr_vs_nih_histograms.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] histograms → {out}")

    print("\nPer-sample stats (mean ± std in normalized space):")
    for label, paths in [("VinDr", v_sample), ("NIH", nih_sample)]:
        for p in paths:
            arr, flipped = _load_normalized(p)
            flip_tag = " [FLIPPED]" if flipped else ""
            print(f"  {label:6s}  {p.name[:35]:35s}  mean={arr.mean():+.3f}"
                  f"  std={arr.std():.3f}{flip_tag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess VinDr-512 with NIH-identical transform")
    p.add_argument("--vindr-dir",     default="data/vindr512/vinbigdata",
                   help="Root of vinbigdata (contains train/ and train.csv)")
    p.add_argument("--out-dir",       default="data/vindr512/std",
                   help="Output root; images go to out-dir/train/")
    p.add_argument("--nih-dir",       default="data/nih/images",
                   help="NIH image directory (for histogram comparison)")
    p.add_argument("--plot-dir",      default="results/vindr_transfer/histograms",
                   help="Where to save histogram and flip-panel PNGs")
    p.add_argument("--mono1-margin",  type=float, default=MONO1_MARGIN,
                   help="lung−heart margin threshold for MONO1 detection (default 40)")
    p.add_argument("--limit",         type=int, default=0,
                   help="Process at most N images (0 = all); useful for smoke-test")
    p.add_argument("--skip-encode",   action="store_true",
                   help="Skip image writing (already done); regenerate plots only")
    p.add_argument("--no-plot",       action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    vindr_root = Path(args.vindr_dir)
    train_dir  = vindr_root / "train"
    csv_path   = vindr_root / "train.csv"

    out_root  = Path(args.out_dir)
    out_train = out_root / "train"
    out_train.mkdir(parents=True, exist_ok=True)

    # ── Label assignment ────────────────────────────────────────────────────
    print("Assigning labels from train.csv …")
    label_map = _assign_labels(csv_path)
    counts = {"normal": 0, "cardio": 0, "effusion": 0, "skip": 0}
    total_in_csv = sum(1 for _ in open(csv_path)) - 1

    all_images = sorted(train_dir.glob("*.png"))
    included: list[tuple[Path, str]] = []
    for img_path in all_images:
        label = label_map.get(img_path.stem)
        if label is None:
            counts["skip"] += 1
        else:
            counts[label] += 1
            included.append((img_path, label))

    print(f"  CSV rows (annotations): {total_in_csv}")
    print(f"  Images on disk:         {len(all_images)}")
    print(f"  Included: normal={counts['normal']}, cardio={counts['cardio']}, "
          f"effusion={counts['effusion']}")
    print(f"  Skipped (other/comorbid): {counts['skip']}")

    if args.limit > 0:
        included = included[: args.limit]
        print(f"  [--limit] processing first {args.limit}")

    manifest_path = out_root / "manifest.csv"

    # ── Skip-encode: plots only ─────────────────────────────────────────────
    if args.skip_encode:
        print("\n[--skip-encode] Skipping image writing, regenerating plots only.")
        manifest_rows: list[dict] = []
        with open(manifest_path, newline="") as f:
            manifest_rows = list(csv.DictReader(f))

        if not args.no_plot:
            nih_pngs = sorted(Path(args.nih_dir).glob("*.png"))
            vindr_paths = [img_path for img_path, _ in included[: len(manifest_rows)]]
            if nih_pngs:
                _plot_histograms(vindr_paths, nih_pngs, Path(args.plot_dir))

            # Flip panel: show top candidates by margin even if below threshold
            margins = []
            for img_path, _ in included:
                arr = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
                m   = _polarity_margin(arr)
                if m > 0:
                    margins.append((m, img_path))
            margins.sort(reverse=True)
            top_paths = [p for _, p in margins[:6]]
            _plot_flip_panel(top_paths, Path(args.plot_dir),
                             margin_thresh=args.mono1_margin)
            n_flagged = sum(1 for m, _ in margins if m > args.mono1_margin)
            print(f"\nPolarity scan (margin > {args.mono1_margin}): "
                  f"{n_flagged} flagged  |  top observed margin = "
                  f"{margins[0][0]:.1f} ({margins[0][1].name[:20]})")
        print("Done.")
        return

    # ── Process images ──────────────────────────────────────────────────────
    print(f"\nProcessing {len(included)} images → {out_train} …")
    print(f"  MONO1 margin threshold: {args.mono1_margin}")

    manifest_rows = []
    n_flipped  = 0
    skipped_io = 0
    flipped_paths: list[Path] = []

    for img_path, label in tqdm(included, unit="img"):
        out_path = out_train / img_path.name
        try:
            arr_norm, flipped = _load_normalized(img_path, args.mono1_margin)
            _save_std(arr_norm, out_path)
        except Exception as e:
            print(f"  SKIP {img_path.name}: {e}", file=sys.stderr)
            skipped_io += 1
            continue
        if flipped:
            n_flipped += 1
            flipped_paths.append(img_path)
        manifest_rows.append({
            "image_id":        img_path.stem,
            "label":           label,
            "label_int":       LABEL_MAP[label],
            "split":           "train",
            "src_path":        str(img_path),
            "out_path":        str(out_path),
            "polarity_flipped": int(flipped),
        })

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "label", "label_int",
                                               "split", "src_path", "out_path",
                                               "polarity_flipped"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nWrote {len(manifest_rows)} entries → {manifest_path}")
    print(f"  MONO1 flipped: {n_flipped}")
    if skipped_io:
        print(f"  IO errors:     {skipped_io}")

    # ── Plots ───────────────────────────────────────────────────────────────
    if not args.no_plot:
        plot_dir  = Path(args.plot_dir)
        nih_pngs  = sorted(Path(args.nih_dir).glob("*.png"))
        if nih_pngs:
            vindr_paths = [Path(r["src_path"]) for r in manifest_rows]
            _plot_histograms(vindr_paths, nih_pngs, plot_dir)
        if flipped_paths:
            _plot_flip_panel(flipped_paths, plot_dir, margin_thresh=args.mono1_margin)
        else:
            # No genuine inversions: show top-N margin candidates as a calibration panel
            margins = []
            for img_path, _ in included:
                arr = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
                m   = _polarity_margin(arr)
                if m > 0:
                    margins.append((m, img_path))
            margins.sort(reverse=True)
            print(f"\nNo MONO1 films detected (threshold={args.mono1_margin}).  "
                  f"Top observed margin: {margins[0][0]:.1f} ({margins[0][1].name[:20]})")
            print("  Generating calibration panel (top-6 by margin, not flipped) …")
            _plot_flip_panel([p for _, p in margins[:6]], plot_dir,
                             margin_thresh=args.mono1_margin)

    print("\nDone.  Next step — encode latents:")
    print(f"  python scripts/precompute_latents.py  (or a VinDr-specific variant)")
    print(f"  --image-dir  {out_train}")


if __name__ == "__main__":
    main()
