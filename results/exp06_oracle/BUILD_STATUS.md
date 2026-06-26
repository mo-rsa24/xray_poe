# Exp 6 — build status & launch handoff

Built 2026-06-26 while the rebalanced-cardio retrain held the GPU. All harness is
written + CPU-smoke-tested; the GPU-bound arms are **deferred to a free GPU** (per the
operational rule: never train/eval concurrently with the retrain — it OOMs the box).

## Done now (GPU-free)
- **Task 3 — patient-leakage guard** ✅ `scripts/exp06_patient_guard.py` → `patient_disjoint_check.md`
  + `data/splits/oracle_both_pool.csv` (557 both-films, assertion `0 shared`).
- **Manifests** ✅ `scripts/exp06_build_manifests.py` → `data/splits/exp06_manifests/`
  (`train_base.csv` 8144, `both_pool.csv` 555, `test.csv` 1413 across 4 strata).
  Label policy: normal[0,0]/cardio_only[1,0]/effusion_only[0,1]; both = CTR/CP-derived
  (cardio=ctr>0.5&border_visible, effusion=either CP blunted). All patient-disjoint
  from the 317 reserved test patients (assertions enforced).

## Built + smoke-tested, awaiting a free GPU
| Task | Script | Produces |
|---|---|---|
| 1 — two-sigmoid detector (CTR labels) | `scripts/train_detector.py --both-n all` | `ckpts/detector_ctr_labels.pt` |
| 2 — Gate 1 pointing-game (Gate 2 = radiologist, pending) | `scripts/exp06_gradcam_pointing.py` | `detector_construct_validity.md` |
| 4 — Oracle arm (abundant, ×3 seeds) | `scripts/train_detector.py --both-n all` | `ckpts/oracle_classifier_s0X.pt`, `oracle_auc.csv` |
| 5 — Starved arm (scarce **N=50**, ×3 seeds) | `scripts/train_detector.py --both-n 50` | `ckpts/starved_classifier_s0X.pt`, `starved_auc.csv` |
| 6 — headroom verdict + figure | `scripts/exp06_headroom_verdict.py` | `verdict.md`, `figures/exp06_oracle_vs_starved_auc.png` |

## To finish (one command, when the GPU is free)
```bash
# preferred — runs the Exp 9 before-model gate (fast) THEN the Exp 6 arms, guarded once:
bash scripts/launch_gpu_batch.sh
#   ONLY_EXP9=1 bash scripts/launch_gpu_batch.sh                                  # just the Exp 9 gate
#   AFTER_LDM=ckpts_rebalanced/model_step0050000.safetensors bash scripts/launch_gpu_batch.sh  # + Exp 9 after-model

# or Exp 6 alone:
bash scripts/launch_exp06.sh      # gated on df disk + a free GPU; runs the Exp 6 sequence in order
```
Both launchers ABORT (exit 1) if the GPU is busy — verified 2026-06-26: with the retrain
live (~21 GB) the combined launcher refused to start and printed the offending PIDs.

## Two human/design notes
- **Gate 2 needs a radiologist** — the pointing-game (Gate 1) is automated; the criterion-
  agreement half is a blinded human sign-off (never CTR again). Recorded as pending.
- **Thin headroom (known-unknown 3).** Oracle pool is only ~555 both-films (506 of 1063
  fell to the reserved test patients); within the 378 both-separable test films the
  instrument labels only ~36 as co-present [1,1] (cardio 286+/effusion 40+). The verdict
  reports this honestly — thin headroom weakens the null's interpretability.
