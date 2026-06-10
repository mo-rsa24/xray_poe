# 🧭 Dataset Acquisition Chain (pre-02)

## Background
The rest of this scope (`01-acquisition…`, `02-integrity…`) currently assumes a
**DICOM** corpus and MONOCHROME1/2 handling. The candidate image-level sources
(NIH ChestX-ray14 = PNG; MIMIC-CXR-JPG, CheXpert = JPG) ship already-rendered,
need no DICOM decoding, and must be storage-fit and hypothesis-aligned (the four
groups: normals, disease-A-alone, disease-B-alone, A+B held-out). This chain
settles the dataset, re-aims the scope, authors the real acquisition tasks, and
reconciles them back into the tree — it runs *before* `02-integrity-manifest`.

## Description
A four-step chain that produces a ready dataset before integrity-scan + EDA:
1. Research the right dataset (four groups, access friction, filtered download).
2. Re-aim this scope's direction off DICOM toward image-level.
3. Author the do-able acquisition tasks + subtasks.
4. Reconcile the new tasks into the plan tree (incl. this MASTER_PLAN), then sync.

## Purpose
Replace the assumed-DICOM acquisition path with a researched, storage-fit,
hypothesis-aligned image-level dataset and a concrete task list — so the data
the experts train on, and the held-out co-morbid set they are tested against,
actually exist locally and fit the disk budget before any modeling.

## Goal
A chosen dataset + access route + filtered-download method; this scope re-aimed
to image-level; acquisition plan files authored; and the plan tree (including
MASTER_PLAN) carrying these chain tasks as checkboxes, synced.

## Tasks
- [x] ✅ Step 1 — Researched the dataset; chose **NIH ChestX-ray14** (~112,120 PNGs, ~42 GB, open access, 1024px frontal-only, four groups via `Data_Entry_2017.csv`). Storage + four-group counts captured in 04.
- [ ] ⚠️ Step 2 — Re-aim this scope off DICOM → image-level via `/init-master-plan`; drop DICOM/monochrome mission + DoD; keep the integrity-scan + manifest end-state intact
- [x] ✅ Step 3 — Authored acquisition tasks via `/populate-plans` → `03-nih-acquisition.md` + `04-four-group-partition.md`, ending where `02-integrity-manifest` begins
- [ ] ⚠️ Step 4 — Reconcile the new tasks into the plans (incl. MASTER_PLAN) via `/goal-reconcile`, then run `/sync-plan-tree`

## Recommended skill
Per-step: `/deep-research` ✅ → `/init-master-plan` ✅ → `/populate-plans` ✅ → `/goal-reconcile` ✅ (then `/sync-plan-tree` ✅). No single skill does the whole chain.

## Engagement Instructions
This plan is "done" when each step's artifact exists. Run the steps in order:

```
# Step 1 — research (cited report naming dataset + access + filtered-download method)
/deep-research which open chest-X-ray datasets (NIH ChestX-ray14, MIMIC-CXR-JPG, CheXpert, PadChest) support image-level cardiomegaly/effusion/pneumothorax labels, normals, and held-out co-morbid cases — and which allow downloading a CSV-filtered subset to fit ~tens of GB

# Step 2 — re-aim direction (MASTER_PLAN.md no longer assumes DICOM/monochrome)
/init-master-plan re-aim the data-foundation master plan for image-level (PNG/JPG) sources with credentialed access and partial download, dropping the DICOM/monochrome assumptions

# Step 3 — author acquisition plans (plans/ holds acquisition + access + filter + four-group partition)
/populate-plans the data-foundation plans: acquisition + access + CSV-filtered partial download + four-group partition, as tasks and subtasks ending where integrity-scan-manifest begins

# Step 4 — reconcile + sync (plan files + MASTER_PLAN carry these as checkboxes)
/goal-reconcile the data-foundation section against its plan files — bring the new acquisition + chain tasks into the plan markdown (and MASTER_PLAN) as checkboxes
/sync-plan-tree
```

Verify the chain is complete:
```
$ ls plans/data-foundation/plans/        # expect new acquisition/access/filter plan files from Step 3
$ grep -i "image-level\|partial download" plans/data-foundation/MASTER_PLAN.md   # Step 2 landed
```
