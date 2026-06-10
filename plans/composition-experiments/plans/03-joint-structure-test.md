# ⭐ Joint-Structure Test (H2 · H2-control · H-anchoring)

## Background
Experiment 6 — the headline.

## Description
Compare PoE-composed vs real both-disease images on the *joint* (heart size, blunting),
across three arms — treatment/`∅`, control/`∅`, treatment/`normal` — all judged vs the floor,
all passed through the VAE.

## Purpose
Only the joint catches a broken correlation; presence can't (both diseases can be present
yet combined unnaturally). This is the paper's central result.

## Goal
Per-arm decisions (gap real / no gap / inconclusive) vs the floor's 95% upper bound, with
corroboration figures.

## Tasks
- [ ] ⚠️ Arm A treatment/`∅` (H2); Arm B control/`∅` (H2-control); Arm C treatment/`normal` (H-anchoring)
- [ ] ⚠️ Two-sample + MMD on the joint (the pair, not each alone), all through the VAE
- [ ] ⚠️ Decide per arm vs the floor; save a (heart size, blunting) scatter, real vs composed

## Engagement Instructions
```
$ python -m experiments.joint --arm treatment --null empty
$ python -m experiments.joint --arm control   --null empty
$ python -m experiments.joint --arm treatment --null normal
# expect e.g. treatment/∅ ~0.78, control/∅ ~0.57 (~floor), treatment/normal ~0.69
# figures/joint_scatter.png
```
