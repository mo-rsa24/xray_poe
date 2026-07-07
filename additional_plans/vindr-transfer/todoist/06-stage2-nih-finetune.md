# Stage-2 NIH Appearance Adaptation

**Description:** Low-LR finetune the Stage-1 ckpt on clean NIH latents + chosen-2nd-disease replay + early-stop.
**Objective:** the NIH-appearance model the Gate-3 eval judges (DoD 6).
**Goal:** a Stage-2 ckpt rendering NIH cardio, the 2nd disease holds, no new leakage.
**Verify:** ckpt loads; 2nd-disease (e.g. effusion) blunt-rate held; leak ≤ 34% floor.

## Tasks
- Decide the adaptation recipe: two-stage finetune (default) vs pooled
  - DATA PREREQUISITE (Gap-A): NIH latents for the chosen 2nd disease must exist — effusion does (data/latents/rebalanced/), pleural thickening does NOT; if 09 picked thickening, build the NIH thickening split + latent cache FIRST
- Plain low-LR finetune on NIH + 2nd-disease/normal replay + early-stop  [resume-init / replay / config]
- Confirm the 2nd disease HOLDS + no new leakage

## Subtasks
- 01-composition-operators
