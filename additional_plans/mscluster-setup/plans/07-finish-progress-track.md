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
- [ ] ⚠️ Run the blunt-check script:
  ```
  python scripts/effusion_blunt_check.py --n 200 --w 1.0
  ```
  Writes results to `results/vae_v2/ldm_sanity/v4_60k_final/`.
- [ ] ⚠️ Remedy (blunt-rate < 0.80): do NOT force a pass. Record the number as-is, keep plan 06
  marked ⚠️, and open a diagnosis task: "effusion blunt-rate below threshold — is this a rendering
  failure or an instrument calibration issue?" Do not close plan 06 until root cause is clear.
- [ ] If blunt-rate ≥0.80: update plan 06 (`additional_plans/vae-v2/plans/06-ldm-retrain.md`)
  body — promote the blunt-rate result and change status to ✅.
- [ ] File DoD-6 in `docs/DECISION_TIMELINE.md`: append a 5-part gate entry (gate name, finding,
  decision, evidence path, date 2026-07-07) following the existing format.
- [ ] Run `/progress-track vae-v2` to update the vae-v2 progress summary.
- [ ] Run `/sync-plan-tree` from the repo root to roll the plan 06 ✅ up to the vae-v2 master plan.

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
