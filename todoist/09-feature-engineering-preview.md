# 09 · Feature-Engineering Preview

## Reference while you do it
- 📄 Plan: plans/eda/plans/07-feature-engineering-preview.md

## Section context (paste into the Todoist section)
**Description:** Preview candidate engineered features derivable from the images before committing — feature ideas + quick distributions. Precursor to the heart-size & blunting extractors in metrics-extractors.
**Objective:** De-risk the feature extractors by previewing what is reliably measurable on this data.
**Goal:** A shortlist of candidate features + quick distributions, flagging those that feed metrics-extractors.
**Verify (whole leaf):** a candidate-feature preview with distributions saved to figures/eda/.
**▶ Recommended prompt (inferred skill — ⚠️ not installed):** `/feature-engineering-preview data/<dataset>` — inferred: surfaces candidate engineered features + quick stats.

## Tasks (one at a time)
- List candidate features measurable from the images (e.g. heart-size proxy, costophrenic-angle/blunting proxy)
- Compute quick distributions for each candidate
- Flag the candidates that will become metrics-extractors extractors
