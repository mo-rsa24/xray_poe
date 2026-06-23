# Sprint — execution order to the 2026-06-27 deadline

Single source of truth for **do-order**. The method-named scopes under `plans/` stay where they
are; this file sequences the work across them and marks what runs in parallel.

Spine = treatment-only paper on existing checkpoints (see `RESULTS_MANIFEST.md`).
Deferred upgrades = `PARKING_LOT.md`. Freeze experiments ~10h before the deadline → write-up.

Status: ⬜ not started · 🔄 in progress · ✅ done

```
S0  KICKOFF                                                        [blocks everything]
         │
    ┌────┴───────────────────────────────────┐
 TRACK A — models (background, longest pole)  TRACK B — eval + paper (foreground)
    │   run A and B in parallel after S0          │
 S1 VAE certification                          S4 Metrics harness + watcher
 S2 Treatment-LDM continuation                 S5 Blinded labels   ← start EARLY (long lead)
                                               S6 Composition evaluation (headline numbers)
                                               S7 Augmentation-utility experiment
                                               S8 Paper writing (as-you-go)
    └────────────────────┬────────────────────┘
 S9  FREEZE + FINALIZE                                             [after A and B converge]
```

## Steps

| # | Step | Track | Scope it lives in | Runs in parallel with | Status |
|---|------|-------|-------------------|------------------------|--------|
| S0 | Kickoff: hippo env up, checkpoints loaded, W&B live | — | (ops) | — | ⬜ |
| S1 | VAE certification: rFID gate → keep or fine-tune | A | `plans/00-vae` | S4 | ⬜ |
| S2 | Treatment-LDM continuation; save checkpoints often | A | `plans/01-single-disease-ldm` | all of Track B | ⬜ |
| S4 | Metrics harness (FID + C2ST + both-present) + checkpoint-watcher + Grad-CAM fix + leakage harness | B | `plans/06-metrics-extractors` | S1, S2 | ⬜ |
| S5 | Blinded radiologist present/absent labels | B | `plans/06-metrics-extractors` (07-visual-leakage) | everything — **start day 0** | ⬜ |
| S6 | Composition evaluation: both-present rate + CIs [N1]; overlay baseline + FID [N2] | B | `plans/07-composition-experiments` | S2, S7 | ⬜ |
| S7 | Augmentation-utility experiment (PoE images → classifier AUC lift) | B | `plans/09-augmentation-utility` (new) | S6 | ⬜ |
| S8 | Paper writing from the figure map, placeholders filled as figures land | B | `plans/08-documentation` + `paper/` | all | ⬜ |
| S9 | Freeze best checkpoint → regenerate all figures → verify hypothesis → polish | — | `plans/08-documentation` | — | ⬜ |

## Parallelism rules

- After **S0**, Track A (S1→S2) and Track B (S4→S6→S7) run **at the same time** on the one GPU.
- The **checkpoint-watcher** (in S4) is the bridge: it auto-runs S4/S6 evals on every checkpoint S2 saves, so Track B re-scores Track A continuously without manual reruns.
- **S5 (labels)** has the longest external lead time — kick it off on day 0, in parallel with everything.
- **S9** is the only hard barrier: it waits for both tracks, then freezes.

## Deferred (not in this sprint — see PARKING_LOT.md)
5-class model + control arm · optimal training grafts · conditional VAE fine-tune (unless S1 fails) ·
three-arm C2ST · Q5 MedSAM / Q6 CTR (instrument-blocked).
