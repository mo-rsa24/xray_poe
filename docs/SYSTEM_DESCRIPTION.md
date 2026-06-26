# System Description — what the system is and does (Methods §0)

> **Purpose.** One source-of-truth statement of the end-to-end system, so the supervisor's three §0 questions are answerable **without reading code**. Grounded in [`scripts/finetune_classifier.py`](../scripts/finetune_classifier.py) (classifier I/O), [`metrics/presence_classifier.py`](../metrics/presence_classifier.py) (classifier inference), [`scripts/ctr_overlay.py`](../scripts/ctr_overlay.py) + [`metrics/eval_checkpoint.py`](../metrics/eval_checkpoint.py) (PSPNet instrument), and the checkpoint inventory in [`EXPERIMENTS.md`](../EXPERIMENTS.md) §6. Authority: [`docs/SUPERVISOR_RECONCILE.md`](SUPERVISOR_RECONCILE.md) (items D1, D9).
>
> **Scope:** proxy task = **cardiomegaly + effusion** on NIH ChestX-ray14. Cluster: hippo. Tracking: W&B.

---

## 1 · The system in one paragraph

We build co-morbid (two-disease) chest X-rays **without any co-morbid training data** by composing two single-disease latent-diffusion experts via Product-of-Experts (PoE), then ask whether those synthetic images make a data-starved disease classifier better. Four models do the work: a **VAE** (the latent codec), an **LDM** (the disease-conditional diffusion generator whose ε-predictions are composed at sampling time), a **DenseNet-121 presence classifier** (the *thing under test* — the downstream model that augmentation is meant to improve), and a **PSPNet anatomical segmenter** (the *instrument* — an independent, classifier-free measurement of cardiothoracic ratio and costophrenic-angle that adjudicates every gate). The classifier is judged; the segmenter judges. They are never the same model, and they are never blurred.

**Dataflow.** Real film → (VAE encode) → latent → (LDM denoise, two disease conditions composed) → synthetic latent → (VAE decode) → synthetic film → (PSPNet instrument gates it on CTR + CP) → if it passes, it joins the classifier's augmentation set → (DenseNet classifier trained, evaluated by stratum on real test films only).

---

## 2 · The classifier under test — exact I/O

The model the whole utility claim is about is a fine-tuned **DenseNet-121** ([`scripts/finetune_classifier.py`](../scripts/finetune_classifier.py), class `PresenceClassifier`; inference in [`metrics/presence_classifier.py`](../metrics/presence_classifier.py)).

**Input pipeline** (identical at train and inference — [`metrics/presence_classifier.py:75`](../metrics/presence_classifier.py#L75) `_preprocess`):

1. Load image as **grayscale** (`PIL` `convert("L")`).
2. **xrv-normalize** to the TorchXRayVision intensity convention: `xrv.datasets.normalize(arr, maxval=255)` → range `[-1024, 1024]`.
3. **Center-crop** to a square: `xrv.datasets.XRayCenterCrop()`.
4. **Resize to 224×224**: `xrv.datasets.XRayResizer(224)`.
5. Result: a `(1, 224, 224)` single-channel tensor.

**Architecture.** `densenet121-res224-nih` backbone (last dense block `denseblock4` + `norm5` unfrozen, everything else frozen) → ReLU → adaptive average pool → **MLP head** (`Linear(feat_dim→256) → ReLU → Dropout(0.4) → Linear(256→2)`).

**Output.** **2 raw logits → sigmoid → 2 independent probabilities `[cardio, effusion]`** — i.e. **multi-label** (each disease scored independently, *not* softmax/mutually-exclusive). Decisions use the **Youden-optimal thresholds computed on the validation set and saved in the checkpoint** (`youden_threshold_cardio`, `youden_threshold_effusion`) — they are *not* hardcoded 0.5. Co-presence on a both-image is declared only when `cardio_prob ≥ thr_c` **and** `effusion_prob ≥ thr_e` simultaneously ([`metrics/presence_classifier.py:297`](../metrics/presence_classifier.py#L297)).

**Training data & labels.** Three single-disease groups only (`images_normal → [0,0]`, `images_cardio_only → [1,0]`, `images_effusion_only → [0,1]`), `BCEWithLogitsLoss` with per-head `pos_weight`, label smoothing 0.1, early stopping on mean val AUC. Reported per disease with **sensitivity + specificity** (the Youden threshold is exactly the sensitivity+specificity-maximizing operating point) alongside AUC.

### 2a · The held-out-both-disease training choice (by design)

**Both-disease images are deliberately excluded from the classifier's training set.** The folder `data/nih/images/` (the co-morbid group) is never seen during fine-tuning — it is held out as the evaluation target ([`scripts/finetune_classifier.py:17`](../scripts/finetune_classifier.py#L17)). Consequence, stated up front: because the classifier learned each disease only from *single-disease* films, it is **expected to struggle on the comorbid (both-present) case** — that struggle is the gap our synthetic augmentation is meant to close, not an accident. This is what makes the downstream test honest: the classifier has no prior exposure to the joint presentation it is evaluated on.

---

## 3 · Full model inventory (with reuse / retrain status)

| Model | Role | Checkpoint | Output | Status (per [`EXPERIMENTS.md`](../EXPERIMENTS.md) §6) |
|---|---|---|---|---|
| **VAE** | latent codec (encode film → latent, decode latent → film) | `ckpts/vae_step0025000.pt` | 4×128×128 latent / reconstructed 512² film | **REUSE** — label-free codec, gated by Exp 2 (codec ceiling found faithful: \|ΔCTR\| median ≈ 0.011). Do **not** retrain. |
| **LDM** | disease-conditional latent diffusion; ε-predictions PoE-composed at sampling | `ckpts/model_step0040000.safetensors` (W&B `prime_lab/paper3-ldm/runs/hifei736`) → **rebalanced-cardio retrain** in `ckpts_rebalanced/` | predicted noise ε(z, condition); composed sample = synthetic latent | **RETRAIN (cardio)** — the `model_step0040000` checkpoint is test-patient-contaminated and under-renders the heart (gen-cardio CTR ≈ 0.395 « real ≈ 0.547); Exp 3 yield NULL fired the Exp 5 rebalanced-cardio retrain. Every synthetic figure is rendered on **both** checkpoints (before/after). VAE + effusion expert reused as-is. |
| **DenseNet-121 presence classifier** | **the model under test** — downstream comorbidity classifier | `ckpts/presence_classifier_finetuned.pt` | 2 sigmoid logits `[cardio, effusion]` (multi-label) + saved Youden thresholds | **EXTEND** — retrain on CTR-derived labels + co-presence data for the Exp 6/7 arms; both-disease held out of training (§2a). |
| **PSPNet anatomical segmenter** | **the instrument** — classifier-free CTR + CP measurement that gates every image | TorchXRayVision ChestX-Det **PSPNet**, pretrained (no local checkpoint; `xrv.baseline_models.chestx_det.PSPNet`) | Heart / Left-Lung / Right-Lung masks → **CTR** (heart width ÷ thorax width) and **CP-angle** (costophrenic wedge, blunted > 80°) | **REUSE** — pretrained, validated by visual sign-off on Exp 1 failure sheets; never trained or fine-tuned here. |

Decoys / do-not-use: `ckpts/ldm/*` is an overfit run; the canonical LDM is `model_step0040000` (pre-retrain) and `ckpts_rebalanced/` (post-retrain).

---

## 4 · What each model outputs (one line each)

- **VAE** → a `4×128×128` latent on encode; a reconstructed `512²` film on decode. It is a **label-free** codec — it carries appearance, not disease identity.
- **LDM** → a predicted-noise vector `ε(z, condition)`; PoE composes two conditions' ε at sampling time (`ε_compose = ε_uncond + w_c·(ε_cardio − ε_uncond) + w_e·(ε_effusion − ε_uncond)`) to yield one synthetic both-disease latent.
- **DenseNet classifier** → two independent sigmoid probabilities `[P(cardio), P(effusion)]`, thresholded by saved Youden values into presence calls. **This is a judgment about disease.**
- **PSPNet instrument** → anatomical masks, reduced to two geometric numbers: **CTR** (> ~0.5 PA ⇒ enlarged heart) and **CP-angle** (blunted ⇒ effusion). **This is a measurement of anatomy, independent of the classifier.**

---

## 5 · Classifier-under-test vs instrument — the disambiguation (do not blur)

These are two different DenseNet/segmentation networks with two different jobs. Confusing them collapses the whole trust chain (independence rule, [`EXPERIMENTS.md`](../EXPERIMENTS.md) §1).

| | **DenseNet-121 classifier** | **PSPNet segmenter** |
|---|---|---|
| Job | The thing **being judged** (downstream utility target) | The thing **doing the judging** (gate instrument) |
| Output | 2 sigmoid logits `[cardio, effusion]`, presence calls | Heart/lung masks → CTR + CP-angle geometry |
| Input | grayscale → xrv-normalize → center-crop → **224×224** | full film → anatomical segmentation |
| Trained here? | Yes — fine-tuned on single-disease NIH groups | No — pretrained ChestX-Det, reused as-is |
| Sees disease labels? | Yes (it predicts them) | No (it segments anatomy only) |

**Independence rule:** CTR *teaches* the cardio label, so CTR (PSPNet) may never also be used to *validate* the classifier on the same axis — the classifier's independent witnesses are GradCAM + radiologist. The PSPNet instrument decides which images are *real/valid co-morbid films*; the DenseNet classifier is then *measured* on that adjudicated set. One never stands in for the other.

---

## 6 · No-leak & sensitivity/specificity (methods commitment — D9)

Two methods-level rules are committed here, before any number is reported, so neither can be relaxed after seeing a result. Both are supervisor requirements (D9 / 7.5, 7.10, 7.1, 7.3). This section is the **statement**; the **computation** lives in plan 11 (`metrics/classifier_report.py`).

### 6a · No leakage — patient-disjoint splits, test on real only

- **Patient-disjoint.** NIH ChestX-ray14 carries **multiple films per patient**, so an image-level split silently leaks: the same patient can land in both train and test. All splits are therefore disjoint **at the patient level**, never the image level. Exp 1 reserves a set of test-patient IDs ([`data/splits/test_patient_ids.txt`](../data/splits/test_patient_ids.txt)); no reserved patient's films may appear in any expert-training or classifier-training split.
- **Test on real only.** The evaluation set is the frozen Exp 1 pixel-verifiable **real** separable-both test films. Synthetic images (PoE or overlay) are **training-side augmentation only** — they never enter the test set. Every reported AUC / sensitivity / specificity is measured on real films the system never trained on.

**Verification path (named, not hand-waved):**
- [`scripts/filter_test_patients.py`](../scripts/filter_test_patients.py) — drops any split film whose `Patient ID` ∈ `test_patient_ids.txt` from the deconfounded splits **in place**, asserts no reserved patient survives, and writes a patient-disjointness report. Idempotent (re-running drops 0).
- [`scripts/cross_leakage_check.py`](../scripts/cross_leakage_check.py) — the *disease*-leakage complement: using the PSPNet instrument, checks the cardio expert does not render effusion (and vice-versa), so "composition" is the only place the two diseases meet.

Together these guard the two leakage axes: **patient leakage** (same patient across splits) and **disease leakage** (one expert rendering the other's disease). Plans 06/07 carry the patient-disjoint assertion into every classifier arm.

### 6b · Sensitivity + specificity everywhere

**Every reported accuracy figure is accompanied by sensitivity and specificity** — accuracy alone hides class imbalance and cannot expose where the classifier fails on comorbidity specifically. The operating point is the **Youden-optimal threshold** (saved per head in the classifier checkpoint, §2), which is by construction the sensitivity+specificity-maximizing point. Sens/spec are reported **per disease and per stratum** (normal / cardio-only / effusion-only / both-separable) alongside AUC, in every results table (Tables 1–2). No accuracy number appears in the paper without its sensitivity/specificity pair. The computation is plan 11 (`metrics/classifier_report.py`).

---

## §0 — answer checklist (three questions → three answers)

**Q1 · What is the system, end-to-end?**
A VAE codec + a disease-conditional LDM whose two single-disease experts are Product-of-Experts–composed (no co-morbid training data) into synthetic both-disease films; a PSPNet instrument gates each film on CTR + CP-angle; the gated films augment a data-starved DenseNet-121 classifier that is evaluated, by stratum, on real test films only. (§1, §3)

**Q2 · What does the classifier take in and put out?**
In: one grayscale film → xrv-normalize → center-crop → **224×224** single-channel tensor. Out: **2 sigmoid logits `[cardio, effusion]`** (multi-label, independent), thresholded by **Youden-optimal values saved in the checkpoint**. Both-disease films are **held out of its training set by design**, so it is expected to struggle on comorbidity — which is the gap the synthetic data targets. (§2, §2a)

**Q3 · Which model is judged and which judges — and what does each output?**
The **DenseNet-121 classifier** is judged (outputs disease probabilities); the **PSPNet segmenter** judges (outputs anatomical masks → CTR + CP geometry, classifier-free). They are distinct networks with distinct inputs and roles and are never interchanged; CTR never both teaches and validates the same model. Full reuse/retrain status: VAE **reuse**, LDM cardio expert **retrain** (rebalanced), classifier **extend**, PSPNet instrument **reuse**. (§3, §4, §5)
