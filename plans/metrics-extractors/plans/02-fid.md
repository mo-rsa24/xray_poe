# 📷 FID

## Description
Implement FID for single-disease and overlay comparisons (used by the Exp4 LDM gate and the
Exp8 floor baseline).

## Purpose
Standard image-realism distance; lower = more realistic. A second angle alongside C2ST/MMD.

## Goal
FID implemented and sanity-checked.

## Tasks
- [ ] ⚠️ Implement FID (Inception features)
- [ ] ⚠️ Sanity-check on real-vs-real (low) and real-vs-noise (high)

## Engagement Instructions
```
$ python -m metrics.fid --a real/ --b real/   # expect near 0
$ python -m metrics.fid --a real/ --b noise/  # expect large
```
