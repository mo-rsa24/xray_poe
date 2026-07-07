# Morbidity Criteria + 2nd-Disease Pick

**Description:** Hand-inspect 10–15 films per morbidity (cardio/effusion/pleural-thickening) with ctr + haze overlays, write criteria, decide the 2nd disease.
**Objective:** gate the 03 class map (DoD 2–3).
**Goal:** a criteria doc + a recorded 2nd-disease decision.
**Verify:** CRITERIA.md per-morbidity rules; second_disease.md names the pick (+ flags the NIH-latent build if thickening wins).
**▶ Recommended prompt:** /visualize-data-samples + /data-integrity-check

## Tasks
- Manual inspection — 10–15 films × {cardiomegaly, pleural effusion, pleural thickening}
- Decide the 2nd disease: pleural effusion vs pleural thickening (VinDr thickening ≈2× effusion boxes; BUT thickening needs a new NIH latent build that effusion already has — weigh both)

## Subtasks
- 01-opacity-instrument
