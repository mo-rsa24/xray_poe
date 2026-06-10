# Composing Two Chest X-ray Diseases by Score Composition

**A first-principles walkthrough: train one conditional LDM on single-disease data, compose two disease conditions at inference via product-of-experts (PoE), and measure whether the result stays ecologically valid.**

This is the whole idea, end to end, in plain English. If you can re-explain each step to a labmate without notes, you've got it.

---

## 1. Definitions

- **Chest X-ray:** a single grayscale image of a chest. May show signs of one disease, both, or none.
- **Cardiomegaly:** enlarged heart — shows up as a heart that takes up too much of the chest width.
- **Pleural effusion:** fluid around the lungs — shows up as blunting at the bottom corners of the lungs (the costophrenic angles).
- **Comorbid / co-occurrence case:** one X-ray that has *both* diseases at once.
- **VAE (autoencoder):** a model that squeezes an image into a small set of numbers (a **latent**) and can rebuild the image from it. It doesn't know about diseases — it just learns to compress and reconstruct chest X-rays.
- **Latent:** the small set of numbers the VAE uses to represent an image. We call it `z`.
- **Diffusion model:** a model that starts from pure noise and removes the noise step by step until a clean image (here, a clean latent `z`) appears.
- **LDM (latent diffusion model):** a diffusion model that does this denoising in the VAE's latent space instead of on raw pixels. Cheaper and sharper.
- **Conditional:** the model takes a label and generates an image *matching that label*. Label = which disease.
- **Condition `c`:** the label fed in. `c1` = cardiomegaly, `c2` = effusion.
- **Noise prediction `ε`:** at each step the model guesses *what noise is currently in the latent*. Subtract that guess, and the latent gets a little cleaner.
- **Score:** the direction that makes the current latent more probable under the model. Mathematically it's the gradient of a log-probability. The noise prediction `ε` is just the score in disguise — same information, different sign and scale.
- **PoE (product of experts):** the composition trick. Run the model twice (once per disease), combine the two noise predictions. Combining noise predictions = **multiplying the two probability distributions**.
- **The joint `p(x | c1, c2)`:** the *true* distribution of real X-rays that have both diseases. This is the target we want to match.
- **Ecological validity:** the generated both-diseases image looks like a *real* both-diseases X-ray — not two diseases crudely pasted together.
- **FID:** a number measuring how close two *sets* of images are as distributions. Lower = more similar.
- **Real-vs-real floor:** the FID you get comparing two halves of the *real* comorbid set to each other. It's the best score achievable at your sample size — your yardstick.

---

## 2. Brief background

Liu et al. (2022) showed you can combine two concepts at generation time by running a diffusion model once per concept and adding their noise predictions. It works well when the concepts are **unrelated** (e.g. "a cat" + "on the left"). Your two diseases are **related** — they often appear together because both come from heart failure. That relatedness is exactly the regime where the trick is expected to strain. Your paper is about whether it strains, by how much, and how to fix it cheaply.

---

## 3. Components

- **One VAE**, trained on all your chest X-rays (any label). Job: turn images into latents and back.
- **One conditional LDM**, trained only on **single-disease** images. It learns "what cardiomegaly-alone looks like" and "what effusion-alone looks like." It never sees a both-diseases image.
- **The PoE composition rule:** the arithmetic that combines the two disease predictions at inference.
- **The held-out comorbid set:** real both-diseases images, kept *out* of training. Used as (a) the yardstick for "did composition produce a realistic joint?" and (b) later, the training signal for the correction.
- **A measurement statistic:** the specific thing you measure to judge realism (e.g. heart width paired with corner-blunting).
- **A residual correction (optional, the constructive contribution):** a small add-on trained on the comorbid set to fix what PoE misses.

---

## 4. Step-by-step pipeline

### Training (done once)

1. Train the VAE on all chest X-rays. Now any image ↔ latent `z`.
2. Train the conditional LDM on single-disease latents only. Feed it `(z, c1)` for cardiomegaly images and `(z, c2)` for effusion images. It learns to denoise toward each disease *on its own*. **It never sees both together.**

### Inference (generate a both-diseases image)

3. Start from a pure-noise latent `z_T`.
4. At each denoising step, run the model **twice** (plus once unconditioned):
   - once conditioned on `c1` → noise prediction `ε₁`
   - once conditioned on `c2` → noise prediction `ε₂`
   - once with no condition → `ε_∅`
5. Combine them:

   ```
   ε_comp = ε_∅ + w₁(ε₁ − ε_∅) + w₂(ε₂ − ε_∅)
   ```

   This combined prediction is the score of the **product** of the two disease distributions.
6. Use `ε_comp` to take one denoising step. Repeat steps 4–6 down to a clean latent `z₀`.
7. Decode `z₀` through the VAE → a generated "both diseases" X-ray.

### Evaluation (the actual science)

8. Generate a large set of composed images.
9. **Presence floor:** check each composed image has both diseases visible (classifier). Necessary, not sufficient.
10. **Joint-structure test:** measure your statistic on composed images *and* on real held-out comorbid images. Compare the two *distributions*. Also compare real-half-vs-real-half to get the floor.
11. **Verdict:** composed distribution ≈ floor → composition reproduced the joint. Composed ≫ floor → there's a real gap.

### (Optional) Closing the gap

12. Show reweighting `w₁, w₂` can't close it (control — proves the gap is structural, not a volume knob).
13. Train a small **residual** model on the comorbid set to predict what PoE missed:

    ```
    ε_corrected = ε_comp + g·ε_residual
    ```

    Re-run evaluation. Plot quality vs. how few comorbid images you used (data-efficiency curve).

---

## 5. Concrete example, traced end to end

Goal: generate one X-ray showing **cardiomegaly + effusion**.

- **Step 1–2 (already trained):** VAE compresses any chest X-ray to a latent, say a `4×64×64` block of numbers. The LDM has seen many big-heart-only and many fluid-only latents, separately.
- **Step 3:** Draw random noise `z_T` — a `4×64×64` block of pure static.
- **Step 4:** Feed `z_T` + "cardiomegaly" → the model says "the noise pointing away from a big heart is `ε₁`." Feed `z_T` + "effusion" → `ε₂`. Feed `z_T` + nothing → `ε_∅`.
- **Step 5:** `ε_comp = ε_∅ + w₁(ε₁−ε_∅) + w₂(ε₂−ε_∅)`. With `w₁=w₂=1`, this nudges the latent toward "big heart" *and* "fluid" at the same time.
- **Step 6:** Subtract a bit of `ε_comp` from `z_T`. The static now leans slightly toward big-heart-and-fluid. Repeat ~50 times.
- **Step 7:** Decode the final clean latent `z₀`. Out comes an X-ray with an enlarged heart *and* blunted corners.
- **The catch (step 10):** measure heart width and corner-blunting on this image. In *real* comorbid X-rays, those two move together in a specific way (cardiac fluid is often bilateral, severity tracks together). Your composed image was built from two experts that **never saw them together**, so the combined image is "biggest-heart-typical" glued to "fluid-typical" — the natural *coupling* between them is missing. The classifier still says "both present" (step 9 passes). The joint-structure test (step 10) is where the mismatch shows.

That mismatch, measured against the real-vs-real floor, is your whole result.

---

## 6. Build each step from first principles

- **VAE (step 1):** encoder network image→`z`, decoder `z`→image. Train to minimize "rebuilt image vs original" plus a term keeping latents well-behaved. Input: images. Output: a working encode/decode pair. No labels needed.
- **Conditional LDM (step 2):** take a U-Net. Inputs: a noised latent, the timestep, and the condition `c` (an embedding). Train it to predict the noise that was added. Only ever pair a latent with its *single* disease label.
- **PoE combine (steps 4–5):** no training. Just call the model three times and do the weighted sum. The only fact you need: *adding noise predictions = multiplying distributions*. That single fact is why this composes concepts and why it assumes the two are independent.
- **Presence classifier (step 9):** standard image classifier trained on real comorbid images to output "cardiomegaly? effusion?" Use it on composed images. Input: image. Output: two yes/no probabilities.
- **Joint-structure test (step 10):** pick a statistic (Section 7). Compute it on composed and on real comorbid images. Compare distributions with a two-sample test (e.g. MMD) or by plotting. Compute the floor from two real halves.
- **Residual correction (step 13):** a small network. Target = (true noise from a real comorbid latent) − (what PoE predicted for it). It learns *only the leftover*. Input: noised comorbid latent + timestep. Output: the correction `ε_residual`.

---

## 7. One small exercise per step

Each is tiny and runnable — do them in order and the whole pipeline becomes real.

1. **VAE:** train a toy autoencoder on ~1000 chest X-rays. Encode one image, decode it, eyeball the reconstruction.
2. **Conditional LDM, no medicine:** build a tiny diffusion model on 2-D points (not images). Condition on two simple labels ("left cluster", "top cluster"), trained *separately*. One forward pass: predict the noise for a noised point.
3. **PoE by hand:** with that toy model, generate points conditioned on "left" and on "top" separately, then with the combined `ε_comp`. Plot all three clouds. See whether the combined cloud lands where left-and-top points *actually* are — or somewhere naïve.
4. **The independence failure, on purpose:** make your two toy labels *correlated* (top points are usually also left). Re-run PoE. Plot composed vs. true top-and-left points. Watch the gap appear. *This is your whole paper in 2-D.*
5. **The floor:** split your real top-and-left points in half, measure the distance between halves. That's the smallest gap you could ever report.
6. **Presence vs. jointness:** train a quick classifier that says "is this point top-and-left?" Confirm it fires happily on composed points that are nonetheless in the *wrong place* — proving presence ≠ realism.
7. **Residual fix:** train a tiny correction on the real top-and-left points, add it to `ε_comp`, re-plot. See the composed cloud move toward the truth. Then redo it with fewer and fewer real points — that's your data-efficiency curve.

---

## The science in one line

PoE composition samples from the **product** of the two single-disease distributions, `p(z|c1)·p(z|c2)`, not the **true joint** `p(z|c1,c2)`. These are equal only when the diseases are conditionally independent. Cardiomegaly and effusion are clinically correlated, so they are not — and your training (single-disease only) plus your hold-out guarantee the correlation enters the model **nowhere**. Measuring that gap against the real-vs-real floor is the experiment.

### Falsifiable hypothesis (H1)

> PoE composition reproduces each disease's **marginal** features but fails to reproduce the **correlation structure** of the comorbid joint.
>
> - (a) *marginals preserved* — per-disease presence on composed images is as high as on real comorbid images;
> - (b) *joint structure broken* — the joint distribution of disease features diverges from real comorbid, beyond the real-vs-real floor.

Falsified if (b) matches the floor as well as real data does.

---

## Two things still thin

- **Your measurement statistic is undefined.** Exercise 4 works in 2-D because the statistic is obvious. On real X-rays you must commit to one — clinical (heart width × corner-blunting) or feature-based (classifier embedding). **H1 is not falsifiable until you pick it.**
- **Whether the data-efficiency curve is the headline or a side ablation** is undecided. It changes what the paper claims.

Everything else is solid enough to start building. Do exercises 1–7 with the toy 2-D model first; they derisk the entire real experiment in an afternoon.
