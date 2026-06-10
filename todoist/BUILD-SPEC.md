# BUILD-SPEC — todoist-publish (Paper 3)

Resumable plan for writing `todoist/` into Todoist. On restart, re-read this and continue
from the last completed item (match by title; re-runs skip existing).

## Status: ✅ COMPLETE — all 39 tasks written (1 section + 7 scope tasks + 32 subtasks)

## Target
- Account: Molefe Molefe · momolefe24@gmail.com (Todoist Free)
- Project: 🧑‍🎓 PhD · `6XHxp94hx7x3PMMR`
- Section: **🩻 PoE Composition of Correlated Chest Diseases** · id = `6gpfrxx3Vqx3wHC2`
- Label: `paper3` · id `2183979036` · color teal
- Scope task IDs: data-foundation `6gpfv4fChpjwjqMR` · eda `6gpfv4h2qgF23MVR` · vae `6gpfv4p7Qx22HPHR` · single-disease-ldm `6gpfv4vwjJGh9xrR` · metrics-extractors `6gpfv55PRc3RgPQR` · composition-experiments `6gpfv56Cjm9xm4xR` · documentation `6gpfv5JFv6238HgR`
- NOTE: EDA expanded 5→7 subtasks live (added 08 Preprocessing /preprocess-vision, 09 Feature-Engineering Preview /feature-engineering-preview) — NOT yet reflected in the on-disk `todoist/` leaf folder or `plans/eda/`.
- Mapping: custom — Section = project; **scope → Task** (master-plan content); **leaf → subtask**
- Priority: p4 · no due dates

## Per-task description templates
- **Scope task:** Description (Mission) / Objective (Objectives) / Goal (Goals) / Done-when (DoD), verbatim-condensed from `plans/<scope>/MASTER_PLAN.md`.
- **Leaf subtask:** Description / Objective / Goal / Verify + "Do (strike through)" bullets, verbatim from `todoist/NN-*.md`.

## Plan (create/skip/update) — order preserved via `order`
| # | Item | Type | Parent | Subtasks | id |
|---|------|------|--------|----------|----|
| — | paper3 | label | — | — | `<pending>` |
| — | 🩻 PoE Composition of Correlated Chest Diseases | section | — | — | `<pending>` |
| 1 | 📦 Data Foundation | task | section | 01,02 | `<pending>` |
| 2 | 🔍 Exploratory Data Analysis | task | section | 03–07 | `<pending>` |
| 3 | 🗜️ VAE | task | section | 08–12 | `<pending>` |
| 4 | 🎲 Single-Disease LDM | task | section | 13–16 | `<pending>` |
| 5 | 📏 Metrics & Extractors | task | section | 17–21 | `<pending>` |
| 6 | 🧪 Composition Experiments | task | section | 22–26 | `<pending>` |
| 7 | 📝 Documentation | task | section | 27–30 | `<pending>` |

Leaf subtasks: the 30 files `todoist/01-*.md … 30-*.md`, each created with `parentId` = its scope task.

## Procedure
1. add-labels paper3 · add-sections section
2. add-tasks: 7 scope tasks (sectionId, label, order 1..7)
3. EYEBALL GATE — create Data Foundation's 2 subtasks, pause for user verification
4. add-tasks: remaining 28 subtasks (parentId per scope, ≤25/call → 2 batches), label, order
5. Report
