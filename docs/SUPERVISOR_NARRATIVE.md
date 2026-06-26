# Supervisor narrative framing (Devon & Richard, 2026-06-26)

Four framing points that refine *how* the results are written up. None changes what we
run; they change emphasis, interpretation, and one caveat (point 3). Each is mapped to
the result we already hold.

---

## 1 · Poor real-comorbidity performance is the PREMISE, not an incidental finding
> Devon: *"that actually needs to be bad because that's the point of our work."*

The classifier failing on real co-occurring disease is the **foundational premise** that
justifies the whole intervention. Without a demonstrated real-comorbidity weakness, there
is no problem for synthetic augmentation to address.

**We have this (Table 1, §7.6), and it is strong:**
- cardio head: 0.827 (cardio-only) → **0.714** (comorbid)
- effusion head: 0.784 (effusion-only) → **0.225** (comorbid)

**Action:** present these *first*, as the load-bearing premise of Section 1 — not buried as
a side table. The effusion collapse (0.784→0.225) is the headline number.

## 2 · Success = comorbidity-SPECIFIC improvement, not general accuracy
> Distinguish "classifier got better overall" from "classifier got better *at comorbidity*."

Synthetic augmentation could lift the single-disease categories while leaving the
co-occurring case untouched — that would **not** support the central claim. The success
criterion is narrow: does augmentation improve the **comorbid stratum** specifically?

**We have this by design:** `section4_table2.py` computes ΔAUC on the **both-separable
(comorbid) stratum**, not overall. 

**Action:** in Table 2, report BOTH the overall ΔAUC *and* the comorbid-stratum ΔAUC
side by side, so the contrast is explicit — a flat comorbid Δ with a positive overall Δ
is still a null for our claim. (Enhance the table to print the overall column.)

## 3 · The label-vs-pixel mismatch is information asymmetry, NOT error — so filtering is STRATIFICATION, not bias-correction
> Radiologists label from multiple projections (AP+PA), longitudinal films, and clinical
> history. A single PA X-ray labelled "cardiomegaly" with a not-obviously-enlarged heart
> is the radiologist *synthesising information we don't have in the image* — not a mistake.

This reframes our excluded-set caveat (D8 / §7.9) and **defends our CTR method**:
- Filtering to visible-cardiac-border films is **selecting the subset where a single PA
  X-ray suffices for diagnosis** — data *stratification*, not bias correction.
- Excluded images are **not bad data** — they carry valid clinical information that is
  simply inaccessible in this modality. Say so explicitly; don't apologise for them.

**This also explains our own numbers.** Real films labelled cardiomegaly sit at median
**CTR ≈ 0.53** (Fig 2) — barely over the 0.5 line — precisely because some labelled-cardio
PA films don't show a big heart in *that* projection. So:
- the CTR instrument judges *"is the heart enlarged in this PA image,"* not *"did the
  radiologist call cardiomegaly"* — and those differ by design (the information gap);
- the real-vs-synth gap we report (real CTR 0.53 vs synth 0.40) is still a genuine
  under-rendering, but the *absolute* 0.5 threshold is soft because of this asymmetry.

**Action:** rewrite D8 (plan 13) from "did we bias by dropping hard cases?" to "we
stratified to the single-PA-diagnosable subset; excluded films hold valid but
out-of-modality information." Quantify the excluded set, but frame it as stratification.

## 4 · Section 1 is problem EXPOSITION, not resolution
> Devon: *"we're not there yet at comorbidities."*

Section 1's job is **diagnostic, not therapeutic** — expose the problem, don't solve it.
The logical arc, in order:
- **(a)** synthetic *individual* diseases generate successfully — *(effusion yes; cardio
  not yet — Exp 9 / Table 1: synth effusion ~0.86, synth cardio 0.20)*
- **(b)** real *comorbidities* are classified poorly — *(point 1: 0.78→0.23)*
- **(c)** synthetic *compositions* fail to improve performance — *(Section 4, expected null)*

Together (a)+(b)+(c) establish **why composition in latent space does not translate to
composition in disease space.** Section 1 = problem statement; Section 4 = does the
proposed fix address it. The paper's arc depends on keeping exposition and
solution-validation distinct.

**Caveat from our own data:** strictly, (a) is only half-true — synthetic *cardio* does
**not** generate successfully (it under-renders the heart). So our honest (a) is "effusion
works, cardio does not yet," which actually *strengthens* the exposure: the breakdown
starts before composition even begins.

---

### What changes in our deliverables
| Point | Already have | Action |
|---|---|---|
| 1 — bad comorbidity = premise | Table 1 §7.6 | lead with it; effusion 0.78→0.23 is the headline |
| 2 — comorbid-specific outcome | `section4_table2.py` (comorbid stratum) | add overall-vs-comorbid contrast column |
| 3 — info asymmetry / stratification | D8 plan 13 (cleaning bias) | **reframe**: stratification not bias; defends CTR |
| 4 — Section 1 = exposition | sober lead (D11) | encode the (a)(b)(c) arc; note cardio caveat |
