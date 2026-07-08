# ✅ Finish Progress Track: effusion blunt-check + close plan 06

## Description
Run `effusion_blunt_check.py` on the Blackwell GPU (this was GPU-blocked on hippo as of 2026-07-07),
record the blunt-rate, then close vae-v2 plan 06, file DoD-6, append the gate to DECISION_TIMELINE.md,
and run `/progress-track vae-v2` + `/sync-plan-tree` to bring the tree current.

## Purpose
O7 finish-progress-track. Plan 06 (`06-ldm-retrain.md`) is the only open item in the vae-v2 scope;
its one pending artifact is the effusion blunt-rate overlay, which needs a GPU. Once recorded, the
vae-v2 scope can be fully closed and the timeline updated.

## Goal
`effusion_blunt_check.py` result recorded (blunt-rate ≥0.80 = pass); plan 06 body updated to ✅;
DoD-6 filed; one gate entry appended to `docs/DECISION_TIMELINE.md`; `/progress-track vae-v2` and
`/sync-plan-tree` both complete.

## Tasks
- [x] ✅ Run the blunt-check script: ✓ ran on Blackwell → `results/vae_v2/ldm_sanity/v4_60k_final/`
  (`effusion_blunt_rate.txt` + `effusion_panel.png`). Had to FIX a bug first: the script read
  `r["effusion_present"]`/`r["cp_max_angle"]` straight off `score_decoded_array`, which returns only
  the geometry dict — those keys come from `_row_from_score`. Unpatched it reported `valid=0/200`,
  `nan%`. Patched to route each score through `EC._row_from_score` (the same two-step the canonical
  `evaluate_checkpoint` uses). Post-fix: **valid=200/200**.
  ```
  python scripts/effusion_blunt_check.py --n 200 --w 1.0
  ```
  Writes results to `results/vae_v2/ldm_sanity/v4_60k_final/`.
- [x] ⚠️ Remedy (blunt-rate < 0.80): TRIGGERED. Measured **blunt-rate = 32.0%** (< 80% mark → FAIL),
  median CP max-angle = 70.5°, valid = 200/200, at w=1.0. Recorded as-is, NOT forced to a pass.
  Plan 06 stays ⚠️. DIAGNOSIS TASK OPENED (needs you): the v4 effusion *classifier* evidence was
  strong (Exp-09 synth sensitivity 0.905→0.960 PASS), yet the *geometric* CP-blunting instrument
  reads only 32% blunted with median angle 70.5° (just under the 80° threshold). Is this (a) a real
  rendering shortfall, (b) a CFG-weight effect — measured at w=1.0 vs the smoke test's w=4.0, or
  (c) an instrument-calibration issue (the 80° cutoff vs a 70.5° median)? Inspect
  `results/vae_v2/ldm_sanity/v4_60k_final/effusion_panel.png` (most-blunted montage) to judge.
  Do not close plan 06 until root cause is clear.
- [ ] ⚠️ PARKED (gate failed): do NOT promote plan 06 to ✅ — blunt-rate 32% < 80%. Blocked on the diagnosis above.
- [ ] ⚠️ PARKED (needs you): file the DoD-6 gate entry in `docs/DECISION_TIMELINE.md` recording the
  32% FAIL honestly (gate name, finding, decision-to-diagnose, evidence path, date). Left for you so the
  timeline narrative reflects your diagnosis conclusion, not a premature close.
- [ ] ⚠️ PARKED (dialogic + gate not passed): `/progress-track vae-v2` — would close plan 06, which is not earned.
- [ ] ⚠️ PARKED: `/sync-plan-tree` — hold until the diagnosis resolves; nothing to roll up as ✅ yet.

## Engagement Instructions
```bash
# 1. Confirm the script wrote output
ls results/vae_v2/ldm_sanity/v4_60k_final/

# 2. Assert blunt-rate >= 0.80 (glob catches whatever filename the script uses)
python -c "
import glob
files = glob.glob('results/vae_v2/ldm_sanity/v4_60k_final/*.json') + \
        glob.glob('results/vae_v2/ldm_sanity/v4_60k_final/blunt*.txt')
assert files, 'no output file found — did effusion_blunt_check.py finish?'
content = open(files[0]).read()
print('output file:', files[0])
print('content:', content[:300])
"

# 3. Confirm plan 06 updated to ✅
grep -E '(✅|blunt)' additional_plans/vae-v2/plans/06-ldm-retrain.md | head -5

# 4. Confirm timeline entry
tail -20 docs/DECISION_TIMELINE.md
```
Expected: output file present; content shows a blunt-rate value ≥0.80; plan 06 contains ✅ and
the blunt-rate; timeline has a gate entry dated 2026-07-07. DoD-7 met — scope closed.

## Recommended skill
▶ `/progress-track vae-v2` ✅ — closes plan 06 DoD and updates the vae-v2 master plan summary.
   alt: `/sync-plan-tree` for rolling the ✅ up the tree.
