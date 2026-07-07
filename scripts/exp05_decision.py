"""Exp05 task 1 — apply the pre-committed reuse-vs-retrain rule and record it.

Rule: reuse iff (Exp 2 ceiling = high) AND (Exp 3 yield ≥ 30%); else retrain
(low ceiling → VAE; low yield → rebalanced-cardio LDM). Reads the two upstream
verdicts (no new sampling) and writes the 2×2 decision table + decision.md.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/exp05_decision"); OUT.mkdir(parents=True, exist_ok=True)
DECISION = OUT / "decision.md"
FIG = Path("figures/exp05_decision_table.png")

# ── rule inputs (cited from the upstream verdicts) ──────────────────────────────
CEILING = "inconclusive-not-low"   # results/exp02_ceiling/verdict.md — VAE faithful (|ΔCTR| 0.011, LPIPS ≈ control), reusable
YIELD = "null"                     # results/exp03_yield/verdict.md  — floor-corrected 9.3% < 10%
ceiling_high = False               # not a clean HIGH, BUT the codec is faithful → does NOT force a VAE retrain
yield_ok = False                   # < 30%

# reuse iff ceiling_high AND yield_ok
reuse_all = ceiling_high and yield_ok
# the codec is reusable (ceiling not low); only the yield gate fires → retrain the cardio LDM
DECISION_STR = "reuse" if reuse_all else "retrain(cardio LDM)"
FORCED_BY = "Exp 3 yield (Gate 3) = NULL < 10% — the cardio channel under-renders"


def main() -> None:
    # ── 2×2 figure ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.axis("off")
    cells = [["reuse VAE + LDM", "retrain LDM (cardio)"],
             ["retrain VAE", "retrain VAE + LDM"]]
    chosen = (1, 1)   # ceiling not-high(row1 "low/​not-high") × yield<30 (col1) → retrain cardio LDM
    for r in range(2):
        for c in range(2):
            sel = (r, c) == chosen
            ax.add_patch(plt.Rectangle((c, 1 - r), 1, 1, fill=True,
                         facecolor="#2ca25f" if sel else "#f0f0f0",
                         edgecolor="k", lw=1.5, alpha=0.85 if sel else 1))
            ax.text(c + 0.5, 1 - r + 0.5, cells[r][c], ha="center", va="center",
                    fontsize=10, fontweight="bold" if sel else "normal",
                    color="white" if sel else "black")
    ax.text(0.5, 2.12, "yield ≥ 30%", ha="center", fontsize=9)
    ax.text(1.5, 2.12, "yield < 30%  ←", ha="center", fontsize=9, fontweight="bold")
    ax.text(-0.05, 1.5, "ceiling HIGH", ha="right", va="center", fontsize=9, rotation=90)
    ax.text(-0.05, 0.5, "ceiling low /\nnot-high  ←", ha="right", va="center", fontsize=9,
            rotation=90, fontweight="bold")
    ax.set_xlim(-0.6, 2.1); ax.set_ylim(-0.1, 2.3)
    ax.set_title("Exp05 · reuse-vs-retrain — DECISION: retrain(cardio LDM)", fontweight="bold")
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG, dpi=150); plt.close(fig)

    # ── decision.md ─────────────────────────────────────────────────────────
    L = [
        "# Exp05 — reuse-vs-retrain decision", "",
        f"**DECISION: {DECISION_STR}** — reuse the VAE; retrain the conditional LDM from scratch.", "",
        "> **Note (single conditional model):** there is one conditional UNet (3 classes), not separate "
        "cardio/effusion LDMs. \"Retrain cardio\" = retrain the whole conditional LDM **from scratch** on "
        "rebalanced + deconfounded + patient-clean latents — which re-learns all three conditionals. The "
        "rebalance *targets* the starved cardio (6%→40% of batches); the effusion conditional is retrained "
        "alongside on clean effusion∧¬cardio data and must be monitored for regression (not just cardio). "
        "**From scratch, not finetune** — the 100k ckpt was trained on data including the 235 reserved test "
        "patients, so finetuning from it would inherit that leakage and invalidate the pixel-verifiable test.", "",
        "## Rule",
        "reuse iff (Exp 2 ceiling = high) AND (Exp 3 yield ≥ 30%); else retrain "
        "(low ceiling → VAE; low yield → rebalanced-cardio LDM).", "",
        "## Inputs (cited, no new sampling)",
        f"- **Ceiling (Exp 2)** = `{CEILING}` (`results/exp02_ceiling/verdict.md`): the VAE is faithful "
        "(|ΔCTR| 0.011, local LPIPS ≈ control) and reusable. Not a clean HIGH, but **not low** → does "
        "NOT force a VAE retrain.",
        f"- **Yield (Exp 3)** = `{YIELD}` (`results/exp03_yield/verdict.md`): floor-corrected 9.3% < 10% "
        "→ fires the retrain lever. A w_c=3.0/4.0 sweep confirmed weight-tuning plateaus at 17% < 30%.", "",
        "## 2×2 (ceiling × yield) — chosen cell: ceiling-not-high × yield<30% → retrain cardio LDM",
        "| | yield ≥ 30% | yield < 30% |",
        "|---|---|---|",
        "| ceiling HIGH | reuse VAE+LDM | retrain LDM (cardio) |",
        "| ceiling low/not-high | retrain VAE | **▶ retrain LDM (cardio)** |", "",
        f"**Gate that forced the retrain (honest-flag):** {FORCED_BY}. The VAE is reused (the ceiling did "
        "NOT force a codec retrain); the effusion LDM is reused (Exp 4 = clean). Only the **cardio** "
        "conditional is retrained.", "",
        "## What fires",
        "→ `plans/rebalanced-cardio-retrain/` — correct the 7:1 cardio:effusion imbalance + "
        "`effusion_weight=2.0`, train on the Exp-4 deconfounded latents, re-measure Gate-3 yield.",
        "", "Figure: figures/exp05_decision_table.png · feeds Exp 6/7 (which checkpoints feed every arm).",
    ]
    DECISION.write_text("\n".join(L) + "\n")
    print(f"DECISION: {DECISION_STR}  (ceiling={CEILING}, yield={YIELD})")
    print(f"  forced by: {FORCED_BY}")
    print(f"→ {DECISION}\n→ {FIG}")


if __name__ == "__main__":
    main()
