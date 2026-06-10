# 26 · Joint-Structure Test (Exp6 / H2 · H2-control · H-anchoring)

## Reference while you do it
- 📄 Plan: plans/composition-experiments/plans/03-joint-structure-test.md

## Section context (paste into the Todoist section)
**Description:** Compare PoE-composed vs real both-disease on the *joint* (heart size, blunting) across three arms — treatment/`∅`, control/`∅`, treatment/`normal` — vs the floor, through the VAE. **The headline.**
**Objective:** Catch a broken correlation (only the joint can — presence can't) and test whether it depends on correlation and on the anchor.
**Goal:** Per-arm decisions vs the floor's 95% upper bound, with corroboration figures.
**Verify (whole leaf):** `python -m experiments.joint --arm {treatment|control} --null {empty|normal}` → e.g. treatment/∅ ~0.78, control/∅ ~0.57 (~floor), treatment/normal ~0.69; figures/joint_scatter.png.

## Tasks (one at a time)
- [ ] Arm A treatment/`∅` (H2); Arm B control/`∅` (H2-control); Arm C treatment/`normal` (H-anchoring)
- [ ] Two-sample + MMD on the joint (the pair, not each alone), all through the VAE
- [ ] Decide per arm vs the floor; save a (heart size, blunting) scatter, real vs composed
