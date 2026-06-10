# 03 · Four-group partition + counts

[⌂ Index](00-INDEX.md) · [← prev 02](02-vindr-acquisition.md) · [next → 04](04-shared-preprocessing.md)

## Reference while you do it
- 📄 Plan: plans/data-foundation/plans/04-four-group-partition.md

## Section context (paste into the Todoist section)
**Description:** Parse the NIH label table and partition the corpus into the four groups the hypothesis needs (normal / cardiomegaly-only / effusion-only / both), then record the counts and on-disk size as a small shared artifact.
**Objective:** Build the actual train/test groups the single-disease experts and the held-out both-disease reference are drawn from, and emit the size fact compute-budget + eda consume.
**Goal:** A partition index (image path → group) plus `data/nih/four_group_counts.md` — the four counts, total on-disk GB, and the dirty-negative caveat.
**Verify (whole leaf):** `python -m data.partition --csv data/nih/Data_Entry_2017.csv --out data/nih/partition.parquet` prints four-group counts (e.g. normal≈60000, cardio_only≈2300, effusion_only≈12000, both≈900); `du -sh data/nih/images` ≈ 42G; `cat data/nih/four_group_counts.md` carries the four counts + GB + the "¬ = not-mentioned" caveat.
**▶ Recommended prompt:** `/data-inventory data/nih` ✅ — auto-profiles size + per-label counts. alt: `/data-distributions` for the label-frequency + co-occurrence view; the partition logic itself is custom.

## Tasks (one at a time)
- [ ] Parse `Data_Entry_2017.csv`; split `Finding Labels` on `|` into a per-image label set
- [ ] Assign each image to a group: `No Finding` → normal; `Cardiomegaly ∧ ¬Effusion`; `Effusion ∧ ¬Cardiomegaly`; both — emit a partition index (path → group)
- [ ] Count each group; measure on-disk GB of the corpus (`du -sh data/nih/images`)
- [ ] Write `data/nih/four_group_counts.md` with the four counts, total GB, and the dirty-negative (`¬` = not-mentioned) caveat — the artifact compute-budget + eda consume
- [ ] Flag if any single-disease group is too thin for a feasibility run (sanity, not a power test)
