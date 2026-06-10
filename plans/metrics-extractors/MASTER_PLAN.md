# 📏 Metrics & Extractors

## Mission
Build and validate the measurement layer every experiment depends on — the
two-sample (C2ST) classifier, MMD, FID, the floor computation, and the
heart-size + pleural-blunting extractors — proven to behave identically on real
and generated images.

## Objectives
1. Implement the two-sample test (C2ST AUC) and MMD distribution distance.
2. Implement FID for single-disease and overlay comparisons.
3. Implement the floor (split real both-disease in half, real-vs-real) with CIs.
4. Build the heart-size and pleural-blunting extractors.
5. Validate the extractors behave identically on real and generated images.

## Goals
1. C2ST + MMD + FID implemented and sanity-checked on known identical / separable pairs.
2. Floor computed with 95% bounds for each pair.
3. Extractor validated: same behavior real vs generated; the coupling it measures named.

## Expected Outcome
A trusted metrics toolkit so the Exp 5–8 numbers are interpretable — especially
the heart-size/blunting extractor, without which Exp6's joint test cannot run.

## Definition of Done
1. C2ST AUC implemented; sanity-checked (≈ 0.5 on identical, ≈ 1.0 on trivially separable).
2. MMD and FID implemented and sanity-checked.
3. Floor computed per pair (real-vs-real split) with 95% upper bound.
4. Heart-size + blunting extractors implemented; the coupling they capture named.
5. Extractor agreement validated on real vs generated (two-sample ≤ threshold);
   validation figure saved.

## Sub-Scopes
(none yet — added by decompose-plan)

## Plans
- ⚠️ 01-two-sample-mmd.md
- ⚠️ 02-fid.md
- ⚠️ 03-floor.md
- ⚠️ 04-extractors.md
- ⚠️ 05-extractor-validation.md
