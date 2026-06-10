# 🗂️ Four-Group Partition + Counts/Size Output

## Background
Runs after `03-nih-acquisition.md` and ends where `02-integrity-manifest.md` begins.
Produces the dataset-size fact that the compute-budget sizing task and the eda
composition task both consume.

## Description
Parse the NIH label table and partition the corpus into the four groups the
hypothesis needs, then record the counts and on-disk size as a small shared
artifact.

1. Labels come from `Data_Entry_2017.csv` — one row per image, `Finding Labels`
   is a pipe-separated string of the positive findings (e.g. `Cardiomegaly|Effusion`).
2. The four groups: normal (`No Finding`), `cardiomegaly ∧ ¬effusion`,
   `effusion ∧ ¬cardiomegaly`, and both.
3. The `¬` here means **the finding was not mentioned**, not that a radiologist
   confirmed it absent — NIH's dirty-negative limit. This caps purity and must be
   recorded alongside the counts so downstream readers see the caveat.

## Purpose
1. Build the actual train/test groups the single-disease experts and the held-out
   both-disease reference are drawn from (Objective 1; feeds the modeling scopes).
2. Emit the four-group counts + on-disk GB as the cross-scope output that
   `compute-budget/plans/01-workload-sizing.md` (dataset-size assumptions) and
   `eda/plans/01-dataset-size-composition.md` (composition table) read from.

## Goal
A partition index (image path → group) plus a small recorded artifact —
`data/nih/four_group_counts.md` (or `.json`) — listing the count in each of the
four groups, the total on-disk GB, and the dirty-negative caveat.

## Tasks
- [ ] ⚠️ Parse `Data_Entry_2017.csv`; split `Finding Labels` on `|` into a per-image label set
- [ ] ⚠️ Assign each image to a group: `No Finding` → normal; `Cardiomegaly ∧ ¬Effusion`; `Effusion ∧ ¬Cardiomegaly`; both — emit a partition index (path → group)
- [ ] ⚠️ Count each group; measure on-disk GB of the corpus (`du -sh data/nih/images`)
- [ ] ⚠️ Write `data/nih/four_group_counts.md` with the four counts, total GB, and the dirty-negative (`¬` = not-mentioned) caveat — the artifact compute-budget + eda consume
- [ ] ⚠️ Flag if any single-disease group is too thin for a feasibility run (sanity, not a power test)

## Recommended skill
▶ `/data-inventory data/nih` ✅ — auto-profiles size and per-label counts for the counts/size half.
   alt: `/data-distributions` for the label-frequency + co-occurrence view; the partition logic itself is custom.

## Engagement Instructions
```
$ python -m data.partition --csv data/nih/Data_Entry_2017.csv --out data/nih/partition.parquet
# expect: prints four-group counts, e.g. normal=~60000, cardio_only=~2300, effusion_only=~12000, both=~900
$ du -sh data/nih/images          # total on-disk size, expect ~42G
$ cat data/nih/four_group_counts.md
# expect: the four counts + total GB + the "¬ = not-mentioned" caveat, ready for compute-budget + eda
```
